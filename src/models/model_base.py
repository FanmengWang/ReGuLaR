import numpy as np
from collections import defaultdict, OrderedDict
from os.path import join as opj
from typing import List
import torch
import lightning.pytorch as pl
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase
from transformers.optimization import get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup
from transformers.models.llama import LlamaForCausalLM
from peft import LoraConfig, get_peft_model

from ..utils.utils import instantiate_from_config, get_timestamp, get_position_ids_from_attention_mask
from ..utils.log import JsonLogger, TextLogger


class LitCoTModelBase(pl.LightningModule):
    def __init__(
        self,
        model_kwargs,
        training_kwargs,
        all_config=None,
    ):
        super().__init__()  # this must be called before save hparams

        self.all_config = all_config
        self.training_kwargs = training_kwargs
        self.model_kwargs = model_kwargs
        self.save_hyperparameters()

        llm_path = opj(all_config.args.workspace_path, "models", "llms", model_kwargs.model_id)
        ### IMPORTANT: replace the llm path to YOUR OWN llm path ###

        # tokenizer
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(llm_path)
        if model_kwargs.get("set_pad_as_last_token", False):  # we don't use this, but might help
            self.tokenizer.pad_token = "[PAD]"
            self.tokenizer.pad_token_id = len(self.tokenizer) - 1
        else:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        # prompt templates
        if model_kwargs.get('chat_template'):
            self.question_template = \
            """<|start_header_id|>system<|end_header_id|>

            Task:
            Think, and then answer a quesiton, split thinkings and answer with ### token.

            """
        else:
            self.question_template = "Question: {} Let's think step by step:"
        self.speed_template = "(Thinking speed: {})"
        self.thinking_separator = "###"
        self.thinking_separator_id = self.tokenizer.convert_tokens_to_ids(self.thinking_separator)
        self.steps_template = "{}"
        self.answer_template = "Answer:{}"

        # llm
        self.llm: LlamaForCausalLM = AutoModelForCausalLM.from_pretrained(llm_path)
        if not model_kwargs.get("set_pad_as_last_token", False):  # not used, but might help
            self.llm.resize_token_embeddings(len(self.tokenizer))
        self.llm.generation_config.pad_token_id = self.tokenizer.pad_token_id
        self.llm.generation_config.eos_token_id = self.tokenizer.eos_token_id
        self.embedding = self.llm.get_input_embeddings()

        # lora (applied after all the configurations readied)
        if model_kwargs.do_lora:
            self.llm = get_peft_model(self.llm, peft_config=LoraConfig(**model_kwargs.lora_config))
            self.llm.print_trainable_parameters()

        # log
        self.sample_logs = defaultdict(dict)

    def configure_optimizers(self):
        kwargs = self.all_config.model.training_kwargs

        self.trainable_parameter_names = []
        trainable_params = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                self.trainable_parameter_names.append(name)
                trainable_params.append(param)

        optimizer = instantiate_from_config(kwargs.optimizer, extra_kwargs={"params": trainable_params})

        if not kwargs.get("use_scheduler", False):
            return {"optimizer": optimizer}
        else:
            scheduler_config = kwargs.scheduler

        if scheduler_config.target == "cosine_schedule_with_warmup":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=scheduler_config.warmup_steps,
                num_training_steps=scheduler_config.num_training_steps,
            )
        else:
            scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=scheduler_config.warmup_steps)

        self.lr_scheduler = scheduler
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    def on_fit_start(self):
        self.text_logger = TextLogger(self, log_file_name="log", tmp_log=self.all_config.args.no_log)
        self.text_logger.log(f"Start training with model:\n {self}\nconfig:\n{self.all_config}")
        self.json_logger = JsonLogger(self, log_file_name="train", tmp_log=self.all_config.args.no_log)
        return super().on_fit_start()

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        log_dict = self.get_log_dict(batch, "train", batch_idx, dataloader_idx=dataloader_idx)
        log_dict.update(self.extra_training_step(batch=batch, batch_idx=batch_idx))
        self.log_dict(log_dict, sync_dist=True, prog_bar=True, batch_size=self.all_config.dataloader.batch_size)
        return log_dict["train/total_loss"]

    def get_log_dict(self, batch, split, batch_idx, dataloader_idx):
        log_dict = self.forward(batch=batch)
        log_dict = {f"{split}/{k}": v for k, v in log_dict.items()}
        return log_dict
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    def extra_training_step(self, batch, batch_idx):
        return {}

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # Evaluate the generation of the model on the validation data
        log_dict = self.eval_generation(batch=batch, split="val", batch_idx=batch_idx, dataloader_idx=dataloader_idx)
        self.log_dict(
            log_dict,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            add_dataloader_idx=False,
            batch_size=len(batch["idx"]),
        )
        return log_dict

    def on_validation_epoch_end(self):
        self.json_logger.log(self.sample_logs)
        return super().on_validation_epoch_end()

    def on_save_checkpoint(self, checkpoint):
        # only save the trainable parameters
        new_state_dict = OrderedDict()
        for k in self.trainable_parameter_names:
            new_state_dict[k] = checkpoint["state_dict"][k]
        checkpoint["state_dict"] = new_state_dict

    def on_test_start(self):
        self.text_logger = TextLogger(self, log_file_name="log")
        self.text_logger.log(f"Start testing with model:\n{self}\nconfig:\n{self.all_config}.")
        self.json_logger = JsonLogger(self, log_file_name=f"test_{get_timestamp()}")
        return super().on_test_start()

    def test_step(self, batch, batch_idx=None, dataloader_idx=0):
        log_dict = self.eval_generation(batch=batch, split="test", batch_idx=batch_idx, dataloader_idx=dataloader_idx)
        self.log_dict(
            log_dict,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            add_dataloader_idx=False,
            batch_size=len(batch["idx"]),
        )

    def on_test_end(self):
        self.json_logger.log(self.sample_logs)
        return super().on_test_end()

    # -- basic methods implemenration ends --#

    # ++ sft implementation begins ++#
    def prepare_inputs(self, text_list, padding_side, part, prefix="", suffix=""):
        if isinstance(text_list, str):
            text_list = [text_list]

        batch_size = len(text_list)
        if isinstance(prefix, str):
            prefix = [prefix] * batch_size
        if isinstance(suffix, str):
            suffix = [suffix] * batch_size

        base_template = getattr(self, f"{part}_template")
        text_list = [prefix[i] + base_template.format(text) + suffix[i] for i, text in enumerate(text_list)]

        inputs = self.tokenizer.batch_encode_plus(
            text_list, return_tensors="pt", add_special_tokens=False, padding="longest", padding_side=padding_side
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        return input_ids, attention_mask
    # -- sft training ends --#

    # ++ evaluation begins ++#
    @torch.no_grad()
    def text_generate(self, questions: List[str]):
        answer_generation_config = self.model_kwargs.answer_generation_config
        batch_size = len(questions)
        # question: [pad, question] or [pad, question, speed, ###]
        input_ids, attention_mask = self.prepare_inputs(
            questions,
            padding_side="left",
            part="question",
            suffix=(self.speed_template.format(1) + self.thinking_separator)
            if self.model_kwargs.sft_method == "cot"
            else "",
        )
        outputs = self.llm.generate(inputs=input_ids, attention_mask=attention_mask, **answer_generation_config)[
            :, input_ids.shape[1] :
        ]
        n_latent_forward = []
        for b in range(batch_size):
            try:
                o = outputs[b].tolist()
                if self.thinking_separator_id in o:
                    length = o.index(self.thinking_separator_id) - 1
                else:
                    length = o.index(self.tokenizer.encode(":", add_special_tokens=False)[0]) - 3
            except ValueError:  # no thinking separator
                length = outputs[b].shape[0]
            n_latent_forward.append(length)
        return outputs, torch.tensor(n_latent_forward, device=self.device, dtype=torch.long).unsqueeze(1)

    @torch.no_grad()
    def latent_generate(
        self,
        questions,
        rl_mode=False,
        return_latent_hidden_states=False,  # for evaluation
    ):
        latent_generation_config = self.model_kwargs.latent_generation_config
        answer_generation_config = self.model_kwargs.answer_generation_config
        max_n_latent_forward = latent_generation_config.max_n_latent_forward
        latent_temperature = latent_generation_config.get("latent_temperature", 1.0)

        batch_size = len(questions)
        n_latent_forward = torch.zeros(size=(batch_size, 1), device=self.device, dtype=torch.long)
        all_inputs_embeds = []

        # 1: question forward
        # question: [pad, question, speed, ###]
        suffix = self.speed_template.format(1) + self.thinking_separator

        question_input_ids, question_attention_mask = self.prepare_inputs(
            questions,
            padding_side="left",
            part="question",
            suffix=suffix,
        )
        question_position_ids = get_position_ids_from_attention_mask(question_attention_mask)
        question_embeds = self.embedding(question_input_ids)
        outputs = self.llm.forward(
            inputs_embeds=question_embeds,
            attention_mask=question_attention_mask,
            position_ids=question_position_ids,
            output_hidden_states=True,
        )
        all_inputs_embeds.append(question_embeds)

        # 2: latent forward
        # 2.1: prepare containers to collect all the intermediate states
        all_attention_mask = question_attention_mask
        current_position_ids = question_position_ids[:, -1:]
        past_key_values = outputs.past_key_values
        is_done = torch.zeros(size=(batch_size, 1), device=self.device, dtype=torch.bool)

        return_latent_inputs_embeds = []
        return_latent_attention_mask = []
        all_latent_hidden_states = []

        # 2.2: latent generation
        for _ in range(max_n_latent_forward):
            if return_latent_hidden_states:
                all_latent_hidden_states.append(torch.stack(outputs.hidden_states, dim=1)[:, :, -1:, :])
            distributions = self.latent_policy.forward(
                outputs.hidden_states[-1][:, -1:, :], temperature=latent_temperature
            )  # outputs from last loops
            current_inputs_embeds = distributions.rsample() * (self.embeds_std)
            return_latent_inputs_embeds.append(current_inputs_embeds)
            all_inputs_embeds.append(current_inputs_embeds)

            not_is_done_long = (~is_done).long()
            all_attention_mask = torch.cat(
                [
                    all_attention_mask,
                    not_is_done_long,
                ],
                dim=1,
            )
            return_latent_attention_mask.append(not_is_done_long)

            current_position_ids = current_position_ids + not_is_done_long
            n_latent_forward += not_is_done_long

            outputs = self.llm.forward(
                inputs_embeds=current_inputs_embeds,
                attention_mask=all_attention_mask,
                position_ids=current_position_ids,
                past_key_values=past_key_values,
                output_hidden_states=True,
            )
            past_key_values = outputs.past_key_values

            last_logits = outputs.logits[:, -1]
            probs = torch.softmax(last_logits / latent_generation_config.get("eol_temperature", 1.0), dim=-1)
            batch_next_token = torch.multinomial(probs, num_samples=1)  # [n, 1]

            is_eol = batch_next_token == self.thinking_separator_id
            is_done = is_done | is_eol
            if is_done.all():
                break

        # 3: add end of thinking
        end_of_thinking_ids = (
            torch.ones(size=(batch_size, 1), device=self.device, dtype=torch.long) * self.thinking_separator_id
        )
        end_of_thinking_embeds = self.embedding(end_of_thinking_ids)
        all_inputs_embeds.append(end_of_thinking_embeds)
        all_attention_mask = torch.cat(
            [
                all_attention_mask,
                torch.ones(size=(batch_size, 1), device=self.device, dtype=torch.long),
            ],
            dim=1,
        )

        # 4: answer generation
        all_inputs_embeds = torch.cat(all_inputs_embeds, dim=1)

        pred_ids = self.llm.generate(
            inputs_embeds=all_inputs_embeds, attention_mask=all_attention_mask, **answer_generation_config
        )

        if rl_mode:
            res = (
                question_input_ids,
                question_attention_mask,
                torch.cat(return_latent_inputs_embeds, dim=1),
                torch.cat(return_latent_attention_mask, dim=1),
                torch.cat(
                    [
                        torch.ones(size=(batch_size, 1), device=self.device, dtype=torch.long)
                        * self.thinking_separator_id,
                        pred_ids,
                    ],
                    dim=1,
                ),  # cat '###' with answer
            )
        elif return_latent_hidden_states:
            res = (
                pred_ids,
                n_latent_forward,
                all_latent_hidden_states,
            )
        else:
            res = (
                pred_ids,
                n_latent_forward,
            )

        return res

    @torch.no_grad()
    def fixed_length_latent_generate(self, questions: List[str]):
        max_n_latent_forward = 6  # this is the hyper-parameter used in Coconut and distill
        answer_generation_config = self.model_kwargs.answer_generation_config
        batch_size = len(questions)
        all_inputs_embeds = []

        # 1: question forward
        # question: [pad, question, speed, ###]
        question_input_ids, attention_mask = self.prepare_inputs(
            questions,
            padding_side="left",
            part="question",
            suffix=self.speed_template.format("auto") + self.thinking_separator,
        )
        question_embeds = self.embedding(question_input_ids)
        outputs = self.llm.forward(
            inputs_embeds=question_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        all_inputs_embeds.append(question_embeds)

        # 2: latent forward
        for i in range(max_n_latent_forward):
            inputs_embeds = outputs.hidden_states[-1][:, -1:, :]  # outputs from last loops
            if hasattr(self, "latent_proj"):
                inputs_embeds = self.latent_proj(inputs_embeds)
            all_inputs_embeds.append(inputs_embeds)
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones(size=(batch_size, 1), device=self.device, dtype=torch.long),
                ],
                dim=1,
            )
            outputs = self.llm.forward(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        # 3: add ### token
        end_of_thinking_ids = (
            torch.ones(size=(batch_size, 1), device=self.device, dtype=torch.long) * self.thinking_separator_id
        )
        end_of_thinking_embeds = self.embedding(end_of_thinking_ids)
        all_inputs_embeds.append(end_of_thinking_embeds)
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(size=(batch_size, 1), device=self.device, dtype=torch.long),
            ],
            dim=1,
        )

        # 4: answer generation
        all_inputs_embeds = torch.cat(all_inputs_embeds, dim=1)
        pred_ids = self.llm.generate(
            inputs_embeds=all_inputs_embeds, attention_mask=attention_mask, **answer_generation_config
        )
        return pred_ids, torch.ones(size=(batch_size, 1), device=self.device, dtype=torch.long) * max_n_latent_forward

    def extract_answer_from_output(self, output_string: str):
        try:
            return output_string.strip('#').split(self.answer_template.format(""))[-1]
        except (ValueError, IndexError):
            return output_string

    def verify_answer(self, gt_answer: str, pred_answer: str) -> float:
        def get_pure_string(s: str):
            return s.strip("#\n ").rstrip(".").replace(",", "").lower()
        gt_answer = get_pure_string(gt_answer)
        pred_answer = get_pure_string(pred_answer)
        try:  # some answers may be like '10.0' but predicted as '10'
            gt_answer = float(gt_answer)
            pred_answer = float(pred_answer)
        except ValueError:
            pass
        return float(gt_answer == pred_answer)

    def eval_generation(self, batch, split="val", batch_idx=None, dataloader_idx=0):
        indices = batch["idx"].tolist()
        questions = batch["question"]
        answers = batch["answer"]
        steps = batch["steps"]

        # predict answers
        if (sft_method := self.model_kwargs.sft_method.lower()) == "regular" or sft_method == "colar":
            outputs_token_ids, n_latent_forward = self.latent_generate(questions=questions)
        elif sft_method == "coconut" or sft_method == "distill":
            outputs_token_ids, n_latent_forward = self.fixed_length_latent_generate(questions=questions)
        elif sft_method == "cot" or sft_method == "icot":
            outputs_token_ids, n_latent_forward = self.text_generate(questions=questions)
        else:
            raise NotImplementedError(f"Unknown sft_method: {sft_method}")

        output_strings = self.tokenizer.batch_decode(outputs_token_ids, skip_special_tokens=True)

        # metric and log
        all_acc = []
        all_output_length = []
        all_latent_forward = []
        for i, q, s, a, o_ids, o_str, nlf in zip(
            indices, questions, steps, answers, outputs_token_ids, output_strings, n_latent_forward
        ):
            if i not in self.sample_logs:
                self.sample_logs[i]["question"] = q
                self.sample_logs[i]["steps"] = s
                self.sample_logs[i]["answer"] = a

                self.sample_logs[i]["pred_answer"] = []
                self.sample_logs[i]["output_string"] = []
                self.sample_logs[i]["output_length"] = []
                self.sample_logs[i]["n_latent_forward"] = []
                self.sample_logs[i]["acc"] = []

            pred_a = self.extract_answer_from_output(o_str)
            acc = self.verify_answer(gt_answer=a, pred_answer=pred_a)
            o_length = (o_ids != self.tokenizer.pad_token_id).sum().item()
            self.sample_logs[i]["pred_answer"].append(pred_a)
            self.sample_logs[i]["output_string"].append(o_str)
            self.sample_logs[i]["output_length"].append(o_length)
            self.sample_logs[i]["n_latent_forward"].append(nlf.item())
            self.sample_logs[i]["acc"].append(acc)

            all_acc.append(acc)
            all_output_length.append(o_length)
            all_latent_forward.append(nlf.item())

        acc_count = sum(all_acc)
        acc_forward_count = sum([a * alf for a, alf in zip(all_acc, all_latent_forward)])
        mean_n_latent_forward_on_acc = np.mean(acc_forward_count / (acc_count + 1e-8))
        mean_acc = np.mean(all_acc)
        mean_n_latent_forward = np.mean(all_latent_forward)
        mean_output_length = np.mean(all_output_length)

        res = {
            "monitor": mean_acc,
            f"{split}/acc": mean_acc,
            f"{split}/n_latent_forward": mean_n_latent_forward,
            f"{split}/n_latent_forward_on_acc": mean_n_latent_forward_on_acc,
            f"{split}/output_length": mean_output_length,
        }
        return res
    # -- evaluation ends --#

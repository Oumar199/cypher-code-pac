from cypher_code_package import (
    pl,
    evaluate,
    LoraConfig,
    TaskType,
    CodeGenForCausalLM,
    torch,
    CodeGenTokenizerFast,
    get_linear_schedule_with_warmup,
    wandb,
    get_peft_model,
)


def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"


class CodeGeneration(pl.LightningModule):

    rouge = evaluate.load("rouge")
    bleu = evaluate.load("sacrebleu")

    def __init__(
        self,
        model_name="Salesforce/codegen-350M-mono",
        model=None,
        lr=1e-4,
        weight_decay=1e-2,
        num_warmup_steps=0,
        num_training_steps=5000,
        r=32,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        max_new_tokens=200,
        predict_with_generate=True,
        splitter="Code Cypher ::",
        padding_side="left",
        num_beams=1,
        use_peft=False,
    ):

        super().__init__()
        
        if model is None:

            self.original_model = CodeGenForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.float32
            )
            
            self.model = model
        
        else:

            self.model = model
        
        if use_peft:
            
            self.lora_config = LoraConfig(
                r=r,  # Rank
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias=bias,
                task_type=TaskType.CAUSAL_LM,  # CodeGen
            )

            self.model = get_peft_model(self.original_model, self.lora_config)

        print(print_number_of_trainable_model_parameters(self.model))

        self.tokenizer = CodeGenTokenizerFast.from_pretrained(
            model_name, padding_side=padding_side
        )

        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.lr = lr

        self.weight_decay = weight_decay

        self.num_warmup_steps = num_warmup_steps

        self.num_training_steps = num_training_steps

        self.predict_with_generate = predict_with_generate

        self.max_new_tokens = max_new_tokens

        self.splitter = splitter

        self.num_beams = num_beams

        self.predictions = {
            "Source references": [],
            "Predictions": [],
            "Target references": [],
        }

    def forward(self, input):

        output = self.model(**input)

        return output.loss, output.logits

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=self.num_training_steps,
        )

        return {'optimizer': optimizer, "lr_scheduler": scheduler}

    def training_step(self, batch, batch_idx=None):

        loss, y_pred = self(batch)

        self.log_dict({"train_loss": loss, 'global_step': float(self.global_step)}, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        wandb.log({"train_loss": loss, "trainer/global_step": self.global_step})

        return loss

    def validation_step(self, batch, batch_idx=None):

        loss, y_pred = self(batch)

        metrics = {}

        if self.predict_with_generate:

            inputs = self.tokenizer.batch_decode(
                batch["input_ids"], skip_special_tokens=True
            )
            references = self.tokenizer.batch_decode(
                batch["input_ids"], skip_special_tokens=True
            )
            
            references = [text.split(self.splitter)[0] + self.splitter if self.splitter in text else text + self.splitter  for text in references]

            inputs = self.tokenizer(references, return_tensors="pt", padding="longest")

            # generate predictions
            predictions = self.model.generate(
                input_ids=inputs.input_ids.to(self.device),
                attention_mask=inputs.attention_mask.to(self.device),
                max_new_tokens=self.max_new_tokens,
            )

            # decode the labels
            predictions = self.tokenizer.batch_decode(
                predictions, skip_special_tokens=True
            )
            labels = self.tokenizer.batch_decode(
                batch["labels"], skip_special_tokens=True
            )

            predictions = [pred.strip().split(self.splitter)[1] if self.splitter in pred else "" for pred in predictions]
        
            labels = [label.strip().split(self.splitter)[1] if self.splitter in label else "" for label in labels]

            # get bleu metric
            bleu = self.bleu.compute(
                predictions=predictions,
                references=[[label.strip()] for label in labels],
            )

            metrics["bleu"] = bleu["score"]

            # get rouge metrics
            rouge = self.rouge.compute(
                predictions=predictions, references=[label.strip() for label in labels]
            )

            metrics.update({k: v for k, v in rouge.items() if "rouge" in k})

        # get the loss
        metrics.update({"eval_loss": loss.item(), 'global_step': float(self.global_step)})

        self.log_dict(metrics, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        metrics.update({"trainer/global_step": self.global_step})

        wandb.log(metrics)

        return loss

    def test_step(self, batch, batch_idx):

        loss, y_pred = self(batch)

        references = self.tokenizer.batch_decode(
            batch["input_ids"], skip_special_tokens=True
        )
        
        references = [text.split(self.splitter)[0] + self.splitter if self.splitter in text else text + self.splitter  for text in references]

        inputs = self.tokenizer(references, return_tensors="pt", padding="longest")

        # generate predictions
        predictions = self.model.generate(
            input_ids=inputs.input_ids.to(self.device),
            attention_mask=inputs.attention_mask.to(self.device),
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            num_beams=self.num_beams
        )

        # decode the labels
        predictions = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = self.tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)
        
        predictions = [pred.strip().split(self.splitter)[1] if self.splitter in pred else "" for pred in predictions]
        
        labels = [label.strip().split(self.splitter)[1] if self.splitter in label else "" for label in labels]

        self.predictions["Source references"].extend(references)
        self.predictions["Predictions"].extend(predictions)
        self.predictions["Target references"].extend(labels)

        # get bleu metric
        bleu = self.bleu.compute(
            predictions=predictions, references=[[label.strip()] for label in labels]
        )

        metrics = {}

        metrics["bleu"] = bleu["score"]

        # get rouge metrics
        rouge = self.rouge.compute(predictions=predictions, references=labels)

        metrics.update({k: v for k, v in rouge.items() if "rouge" in k})

        # get the loss
        metrics.update({"test_loss": loss.item(), 'global_step': float(self.global_step)})

        # log metrics
        self.log_dict(metrics, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

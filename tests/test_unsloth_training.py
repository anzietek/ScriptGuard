import sys
from unittest.mock import MagicMock

sys.modules['unsloth'] = MagicMock()
sys.modules['unsloth.FastLanguageModel'] = MagicMock()
sys.modules['unsloth.UnslothTrainer'] = MagicMock()

import pytest
from unittest.mock import Mock, patch
from datasets import Dataset
from scriptguard.models.qlora_finetuner import QLoRAFineTuner


@pytest.fixture
def mock_dataset() -> Dataset:
    """Create a mock dataset for testing."""
    data = {
        "text": [
            "import os\nos.system('echo hello')",
            "def benign_function():\n    return 42",
            "import subprocess\nsubprocess.call(['ls'])"
        ]
    }
    return Dataset.from_dict(data)


@pytest.fixture
def mock_config() -> dict:
    """Create a mock configuration dictionary."""
    return {
        "training": {
            "tokenizer_max_length": 512,
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
            "batch_size": 2,
            "gradient_accumulation_steps": 4,
            "num_epochs": 1,
            "learning_rate": 2e-4,
            "weight_decay": 0.01,
            "warmup_steps": 10,
            "lr_scheduler_type": "linear",
            "fp16": False,
            "bf16": True,
            "optim": "adamw_8bit",
            "logging_steps": 5,
            "evaluation_strategy": "no",
            "save_steps": 100,
        }
    }


class TestQLoRAFineTuner:
    """Test suite for QLoRAFineTuner with unsloth integration."""

    def test_initialization(self, mock_config: dict) -> None:
        """Test QLoRAFineTuner initialization."""
        finetuner = QLoRAFineTuner(
            model_id="bigcode/starcoder2-3b",
            config=mock_config
        )

        assert finetuner.model_id == "bigcode/starcoder2-3b"
        assert finetuner.config == mock_config
        assert finetuner.model is None
        assert finetuner.tokenizer is None

    def test_initialization_default_config(self) -> None:
        """Test QLoRAFineTuner initialization with default config."""
        finetuner = QLoRAFineTuner(model_id="test/model")

        assert finetuner.model_id == "test/model"
        assert finetuner.config == {}
        assert finetuner.model is None
        assert finetuner.tokenizer is None

    @patch('scriptguard.models.qlora_finetuner.FastLanguageModel')
    @patch('scriptguard.models.qlora_finetuner.UnslothTrainer')
    @patch('scriptguard.models.qlora_finetuner.torch')
    def test_train_basic(
        self,
        mock_torch: Mock,
        mock_trainer_class: Mock,
        mock_flm: Mock,
        mock_dataset: Dataset,
        mock_config: dict
    ) -> None:
        """Test basic training flow with unsloth."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.is_bf16_supported.return_value = True

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": [[1, 2, 3]],
            "attention_mask": [[1, 1, 1]]
        }

        mock_flm.from_pretrained.return_value = (mock_model, mock_tokenizer)
        mock_flm.get_peft_model.return_value = mock_model

        mock_trainer = MagicMock()
        mock_trainer_class.return_value = mock_trainer

        finetuner = QLoRAFineTuner(
            model_id="bigcode/starcoder2-3b",
            config=mock_config
        )

        finetuner.train(mock_dataset, output_dir="./test_output")

        mock_flm.from_pretrained.assert_called_once_with(
            model_name="bigcode/starcoder2-3b",
            max_seq_length=512,
            dtype=None,
            load_in_4bit=True,
        )

        mock_flm.get_peft_model.assert_called_once()
        mock_trainer.train.assert_called_once()
        mock_model.save_pretrained.assert_called_once_with("./test_output/final_adapter")
        mock_tokenizer.save_pretrained.assert_called_once_with("./test_output/final_adapter")

    @patch('scriptguard.models.qlora_finetuner.FastLanguageModel')
    @patch('scriptguard.models.qlora_finetuner.UnslothTrainer')
    @patch('scriptguard.models.qlora_finetuner.torch')
    def test_train_with_eval_dataset(
        self,
        mock_torch: Mock,
        mock_trainer_class: Mock,
        mock_flm: Mock,
        mock_dataset: Dataset,
        mock_config: dict
    ) -> None:
        """Test training with evaluation dataset."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.is_bf16_supported.return_value = True

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": [[1, 2, 3]],
            "attention_mask": [[1, 1, 1]]
        }

        mock_flm.from_pretrained.return_value = (mock_model, mock_tokenizer)
        mock_flm.get_peft_model.return_value = mock_model

        mock_trainer = MagicMock()
        mock_trainer_class.return_value = mock_trainer

        mock_config["training"]["evaluation_strategy"] = "steps"
        mock_config["training"]["eval_steps"] = 50

        eval_dataset = Dataset.from_dict({"text": ["test code"]})

        finetuner = QLoRAFineTuner(
            model_id="bigcode/starcoder2-3b",
            config=mock_config
        )

        finetuner.train(mock_dataset, eval_dataset=eval_dataset, output_dir="./test_output")

        mock_trainer.train.assert_called_once()

        trainer_call_args = mock_trainer_class.call_args
        assert trainer_call_args.kwargs["eval_dataset"] is not None

    @patch('scriptguard.models.qlora_finetuner.FastLanguageModel')
    @patch('scriptguard.models.qlora_finetuner.UnslothTrainer')
    @patch('scriptguard.models.qlora_finetuner.torch')
    def test_train_cpu_fallback(
        self,
        mock_torch: Mock,
        mock_trainer_class: Mock,
        mock_flm: Mock,
        mock_dataset: Dataset,
        mock_config: dict
    ) -> None:
        """Test training falls back to CPU when CUDA unavailable."""
        mock_torch.cuda.is_available.return_value = False

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": [[1, 2, 3]],
            "attention_mask": [[1, 1, 1]]
        }

        mock_flm.from_pretrained.return_value = (mock_model, mock_tokenizer)
        mock_flm.get_peft_model.return_value = mock_model

        mock_trainer = MagicMock()
        mock_trainer_class.return_value = mock_trainer

        finetuner = QLoRAFineTuner(
            model_id="bigcode/starcoder2-3b",
            config=mock_config
        )

        finetuner.train(mock_dataset, output_dir="./test_output")

        trainer_call_args = mock_trainer_class.call_args
        training_args = trainer_call_args.kwargs["args"]
        assert training_args.fp16 is False
        assert training_args.bf16 is False

    @patch('scriptguard.models.qlora_finetuner.FastLanguageModel')
    @patch('scriptguard.models.qlora_finetuner.UnslothTrainer')
    @patch('scriptguard.models.qlora_finetuner.torch')
    def test_train_bf16_fallback(
        self,
        mock_torch: Mock,
        mock_trainer_class: Mock,
        mock_flm: Mock,
        mock_dataset: Dataset,
        mock_config: dict
    ) -> None:
        """Test training falls back from BF16 to FP16 when unsupported."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.is_bf16_supported.return_value = False

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": [[1, 2, 3]],
            "attention_mask": [[1, 1, 1]]
        }

        mock_flm.from_pretrained.return_value = (mock_model, mock_tokenizer)
        mock_flm.get_peft_model.return_value = mock_model

        mock_trainer = MagicMock()
        mock_trainer_class.return_value = mock_trainer

        finetuner = QLoRAFineTuner(
            model_id="bigcode/starcoder2-3b",
            config=mock_config
        )

        finetuner.train(mock_dataset, output_dir="./test_output")

        trainer_call_args = mock_trainer_class.call_args
        training_args = trainer_call_args.kwargs["args"]
        assert training_args.fp16 is True
        assert training_args.bf16 is False

    @patch('scriptguard.models.qlora_finetuner.FastLanguageModel')
    @patch('scriptguard.models.qlora_finetuner.UnslothTrainer')
    @patch('scriptguard.models.qlora_finetuner.torch')
    def test_lora_config_from_yaml(
        self,
        mock_torch: Mock,
        mock_trainer_class: Mock,
        mock_flm: Mock,
        mock_dataset: Dataset,
        mock_config: dict
    ) -> None:
        """Test LoRA configuration is correctly applied from config.yaml."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.is_bf16_supported.return_value = True

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": [[1, 2, 3]],
            "attention_mask": [[1, 1, 1]]
        }

        mock_flm.from_pretrained.return_value = (mock_model, mock_tokenizer)
        mock_flm.get_peft_model.return_value = mock_model

        mock_trainer = MagicMock()
        mock_trainer_class.return_value = mock_trainer

        mock_config["training"]["lora_r"] = 32
        mock_config["training"]["lora_alpha"] = 64
        mock_config["training"]["lora_dropout"] = 0.1
        mock_config["training"]["target_modules"] = ["q_proj", "k_proj"]

        finetuner = QLoRAFineTuner(
            model_id="bigcode/starcoder2-3b",
            config=mock_config
        )

        finetuner.train(mock_dataset, output_dir="./test_output")

        peft_call_args = mock_flm.get_peft_model.call_args
        assert peft_call_args.kwargs["r"] == 32
        assert peft_call_args.kwargs["lora_alpha"] == 64
        assert peft_call_args.kwargs["lora_dropout"] == 0.1
        assert peft_call_args.kwargs["target_modules"] == ["q_proj", "k_proj"]

    @patch('scriptguard.models.qlora_finetuner.FastLanguageModel')
    @patch('scriptguard.models.qlora_finetuner.UnslothTrainer')
    @patch('scriptguard.models.qlora_finetuner.torch')
    def test_tokenization_parameters(
        self,
        mock_torch: Mock,
        mock_trainer_class: Mock,
        mock_flm: Mock,
        mock_dataset: Dataset,
        mock_config: dict
    ) -> None:
        """Test tokenization uses correct parameters."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.is_bf16_supported.return_value = True

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": [[1, 2, 3]],
            "attention_mask": [[1, 1, 1]]
        }

        mock_flm.from_pretrained.return_value = (mock_model, mock_tokenizer)
        mock_flm.get_peft_model.return_value = mock_model

        mock_trainer = MagicMock()
        mock_trainer_class.return_value = mock_trainer

        mock_config["training"]["tokenizer_max_length"] = 1024

        finetuner = QLoRAFineTuner(
            model_id="bigcode/starcoder2-3b",
            config=mock_config
        )

        finetuner.train(mock_dataset, output_dir="./test_output")

        from_pretrained_call = mock_flm.from_pretrained.call_args
        assert from_pretrained_call.kwargs["max_seq_length"] == 1024

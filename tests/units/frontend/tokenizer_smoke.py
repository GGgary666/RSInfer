"""Optional HF adapter smoke test using a locally trained tiny tokenizer, no downloads.

PYTHONPATH=engine:. uv run --extra frontend python -m tests.units.frontend.tokenizer_smoke
"""

from __future__ import annotations

import importlib
import sys
from tempfile import TemporaryDirectory

from frontend.detok.unit import Detokenizer
from frontend.interfaces import Message
from frontend.tokenizer.unit import HFTokenizer


def main() -> None:
    tokenizers = importlib.import_module("tokenizers")
    transformers = importlib.import_module("transformers")
    with TemporaryDirectory() as directory:
        backend = tokenizers.Tokenizer(tokenizers.models.BPE(unk_token="<unk>"))
        backend.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=False)
        backend.decoder = tokenizers.decoders.ByteLevel()
        backend.train_from_iterator(["中文 hello!", "user: hi assistant:"], tokenizers.trainers.BpeTrainer(
            vocab_size=280, special_tokens=["<unk>", "<eos>"],
            initial_alphabet=tokenizers.pre_tokenizers.ByteLevel.alphabet()))
        tokenizer = transformers.PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="<eos>",
                                                         unk_token="<unk>")
        tokenizer.chat_template = ("{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}"
                                   "{% if add_generation_prompt %}assistant: {% endif %}")
        tokenizer.save_pretrained(directory)
        service = HFTokenizer(directory)
        assert service.chat([Message("user", "hi")])
        stream = Detokenizer(service).open()
        text = "".join(stream.push([token]).text for token in service.encode("中文 hello!"))
        text += stream.push(list(service.stop_token_ids), final=True).text
        assert text == "中文 hello!", repr(text)
        print(f"HF tokenizer/chat template/incremental Unicode: passed; torch loaded: {'torch' in sys.modules}")


if __name__ == "__main__":
    main()

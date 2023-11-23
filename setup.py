from transformers import LayoutLMForTokenClassification, LayoutLMv2Processor, LayoutLMv2TokenizerFast


tokenizer = LayoutLMv2TokenizerFast.from_pretrained("microsoft/layoutlmv2-base-uncased")
model = LayoutLMForTokenClassification.from_pretrained("trng1305/layoutlmv2-sroie-test")
processor = LayoutLMv2Processor.from_pretrained("trng1305/layoutlmv2-sroie-test")

tokenizer.save_pretrained("./my-tokenizer")
processor.save_pretrained("./my-processor")
model.save_pretrained("./my-model")


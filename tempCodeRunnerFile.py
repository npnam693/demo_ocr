
tokenizer = LayoutLMv2TokenizerFast.from_pretrained("./my-tokenizer")
model = LayoutLMForTokenClassification.from_pretrained("./my-model")
processor = LayoutLMv2Processor.from_pretrained("./my-processor")
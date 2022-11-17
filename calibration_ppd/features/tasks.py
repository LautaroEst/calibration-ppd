
from ..core import Task
import transformers


class LoadPretrainedTokenizer(Task):

    def __init__(self,model,source):
        self.model = getattr(transformers,model)
        self.source = source

    def run(self):
        tokenizer = self.model.from_pretrained(self.source)
        return tokenizer

    def save(self,output,output_dir):
        output.save_pretrained(output_dir)

    def load(self,output_dir):
        tokenizer = self.model.from_pretrained(output_dir)
        return tokenizer
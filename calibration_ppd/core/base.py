

import pickle
import os
from typing import Dict


class Task:

    def run(self,**inputs) -> Dict:
        raise NotImplementedError

    def save_output_to_disk(self,output):
        try:
            self.save(output,self.get_output_dir())
        except NotImplementedError:
            with open(os.path.join(self.get_output_dir(),"output.pkl"),"wb") as f:
                pickle.dump(output,f)

    def load_output_from_disk(self):
        try:
            output = self.load(self.get_output_dir())
        except NotImplementedError:
            with open(os.path.join(self.get_output_dir(),"output.pkl"),"rb") as f:
                output = pickle.load(f)
        return output

    def save(self,output,output_dir):
        raise NotImplementedError

    def load(self,output_dir):
        raise NotImplementedError

    def get_output_dir(self):
        return self._output_dir

    def set_output_dir(self,output_dir):
        self._output_dir = output_dir

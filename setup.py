from setuptools import setup

setup(name="cypher_codegen", version="0.0.1", author="Oumar Kane", author_email="oumar.kane@univ-thies.sn", 
      description="Contain function and classes to tune and fine tune codegen model.",
      install_requires=['transformers', 'evaluate', 'pytorch-lightning==1.9.0', 'torch', 'rouge_score', 'sacrebleu', 'loralib', 'peft', 'pandas', 'numpy', 'wandb', 'matplotlib'])

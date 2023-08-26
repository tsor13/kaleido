# Value Kaleidoscope

This repository is a companion for [Value Kaleidoscope: Engaging AI with Pluralistic Values, Rights, and Duties](https://kaleido.allen.ai). Here are some other links of interest: [demo](https://kaleido.allen.ai), [dataset](https://huggingface.co/datasets/tsor13/ValuePrism), and models ([small](https://huggingface.co/tsor13/kaleido-small), [base](https://huggingface.co/tsor13/kaleido-base), [large](https://huggingface.co/tsor13/kaleido-large), [xl](https://huggingface.co/tsor13/kaleido-xl), [xxl](https://huggingface.co/tsor13/kaleido-xxl)).

## Example usage

To intiialize the system, you can use the following code:

```python
from KaleidoSys import KaleidoSys
system = KaleidoSys('tsor13/kaleido-small')
```

From here, you can use the system to generate a candidate set of values, rights, and duties:

```python
system.get_candidates('Biking to work instead of driving')
```
Output:
```python
{'duties': ['to not drive to work'],
 'rights': ['to bike to work'],
 'values': ['to reduce carbon emissions']}
```

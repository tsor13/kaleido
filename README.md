# Value Kaleidoscope

This repository is a companion for [Value Kaleidoscope: Engaging AI with Pluralistic Values, Rights, and Duties](https://kaleido.allen.ai). Here are some other links of interest: [demo](https://kaleido.allen.ai), [dataset](https://huggingface.co/datasets/tsor13/ValuePrism), and models ([small](https://huggingface.co/tsor13/kaleido-small), [base](https://huggingface.co/tsor13/kaleido-base), [large](https://huggingface.co/tsor13/kaleido-large), [xl](https://huggingface.co/tsor13/kaleido-xl), [xxl](https://huggingface.co/tsor13/kaleido-xxl)).

## Example usage

To intiialize the system, you can use the following code:

```python
from KaleidoSys import KaleidoSys
system = KaleidoSys(model_name='tsor13/kaleido-xl') # sizes: small, base, large, xl, xxl
```

#### Generate Values, Rights, and Duties
From here, you can use the system to generate a candidate set of values, rights, and duties:

```python
system.get_candidates('Biking to work instead of driving')
```
Output:
```python
                              action    vrd                                         value  relevant  supports  opposes  either     label
0  Biking to work instead of driving   Duty        Duty to be environmentally responsible      1.00      1.00     0.00    0.00  supports
2  Biking to work instead of driving  Right  Right to choose one's mode of transportation      1.00      0.26     0.00    0.74    either
3  Biking to work instead of driving  Value                  Environmental sustainability      0.99      1.00     0.00    0.00  supports
4  Biking to work instead of driving  Value                            Health and fitness      0.99      1.00     0.00    0.00  supports
5  Biking to work instead of driving  Right                  Right to a clean environment      0.99      1.00     0.00    0.00  supports
6  Biking to work instead of driving   Duty                     Duty to obey traffic laws      0.99      0.13     0.01    0.86    either
7  Biking to work instead of driving  Value                                   Convenience      0.98      0.03     0.75    0.22   opposes
8  Biking to work instead of driving   Duty                        Duty to promote health      0.98      1.00     0.00    0.00  supports
```

#### Evaluate Relevance
```python
system.get_relevance('Do a pushup', 'Value', 'Health')
```
Output:
```python
tensor([0.9975, 0.0025]) # first number is p(relevant), second is p(not relevant)
```

#### Evaluate Valence
```python
system.get_valence('Do a pushup', 'Value', 'Health')
```
Output:
```python
tensor([9.9961e-01, 6.0631e-06, 3.8288e-04]) # p(Supports), p(Opposes), p(Either)
```

#### Explanation
```python
system.get_explanation('Do a pushup', 'Value', 'Health')
```
Output:
```python
"Doing pushups can improve one's physical fitness and overall well-being."
```

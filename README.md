# Value Kaleidoscope

This repository is a companion for [Value Kaleidoscope: Engaging AI with Pluralistic Values, Rights, and Duties](https://kaleido.allen.ai). Here are some other links of interest: [demo](https://kaleido.allen.ai), [dataset](https://huggingface.co/datasets/tsor13/ValuePrism), and models ([small](https://huggingface.co/tsor13/kaleido-small), [base](https://huggingface.co/tsor13/kaleido-base), [large](https://huggingface.co/tsor13/kaleido-large), [xl](https://huggingface.co/tsor13/kaleido-xl), [xxl](https://huggingface.co/tsor13/kaleido-xxl)).

## Example usage

To intiialize the system, you can use the following code:

```python
from KaleidoSys import KaleidoSys
system = KaleidoSys(model_name='tsor13/kaleido-small') # sizes: small, base, large, xl, xxl
```

#### Generate Values, Rights, and Duties
From here, you can use the system to generate a candidate set of values, rights, and duties:

```python
system.get_candidates('Biking to work instead of driving')
```
Output:
```python
                               action    vrd                                         value  relevant  supports  opposes  either    label
0   Biking to work instead of driving   Duty   Duty to maintain a healthy work environment      0.98      0.15     0.74    0.11  opposes
1   Biking to work instead of driving  Right           Right to a healthy work environment      0.97      0.17     0.76    0.07  opposes
2   Biking to work instead of driving   Duty                 Duty to abide by traffic laws      0.95      0.07     0.88    0.05  opposes
3   Biking to work instead of driving   Duty  Duty to be considerate of others' well-being      0.94      0.10     0.76    0.14  opposes
4   Biking to work instead of driving  Value                             Work-life balance      0.93      0.10     0.83    0.07  opposes
6   Biking to work instead of driving  Value                           Personal well-being      0.85      0.21     0.70    0.09  opposes
9   Biking to work instead of driving  Value                                        Safety      0.82      0.15     0.81    0.04  opposes
10  Biking to work instead of driving  Value                                    Efficiency      0.76      0.13     0.81    0.06  opposes
11  Biking to work instead of driving  Value                                Responsibility      0.71      0.16     0.78    0.06  opposes
```

#### Evaluate Relevance
```python
system.get_relevance('Do a pushup', 'Value', 'Health')
```
Output:
```python
tensor([0.9310, 0.0690]) # first number is p(relevant), second is p(not relevant)
```

#### Evaluate Valence
```python
system.get_valence('Do a pushup', 'Value', 'Health')
```
Output:
```python
tensor([0.6247, 0.2245, 0.1508]) # p(Supports), p(Opposes), p(Either)
```

#### Explanation
```python
system.get_explanation('Do a pushup', 'Value', 'Health')
```
Output:
```python
'Pushups can improve the health of the individual, promoting their well-being and promoting overall well-being.'
```

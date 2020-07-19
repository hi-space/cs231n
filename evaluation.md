---
description: 'Precision, Recall, Accuracy'
---

# Evaluation

![Confusion Matrix](.gitbook/assets/image%20%28454%29.png)

* 정답을 맞추면 True
* 오답이면 False
* True 라고 예측하면 Positive
* False라고 예측하면 Negative

### Precision \(Positive Predictive Value\)

$$
Preccision = \frac{TP} {TP + FP}
$$

모델이 True 라고 분류한 것 중 실제 True인 것의 비율

### Recall

$$
Recall = \frac{TP}{TP+FN}
$$

실제 정답이 True 인 것 중에서 모델이 True 라고 예측한 것의 비율

### Accuracy

$$
Accuracy = \frac{TP+TN}{TP+FN+FP+TN}
$$

Precision과 Recall은 True인 것에 대한 비율이였다. Accuracy는 False인 것을 맞춘 경우도 고려한 비율이다. 직관적으로 모델의 성능을 나타낼 수 있는 평가 지표이다.

## Reference

[https://sumniya.tistory.com/26](https://sumniya.tistory.com/26)


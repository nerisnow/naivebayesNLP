import re #for splitting
import csv


from collections import Counter

with open("train.csv", 'r') as file:
  trainset = list(csv.reader(file))

def get_text(reviews, score):
  # Join together the text in the reviews for a particular tone(positive or negative tone)
  #  making all lowercase to avoid "Not" and "not" being seen as different words, for example.
  return " ".join([r[0].lower() for r in trainset if r[1] == str(score)])

def count_text(text):
  # Split text into words based on whitespace
  words = re.split("\s+", text)
  # Count up the occurence of each word and store as a dictionary data type
  return Counter(words)

negative_text = get_text(trainset, 0)
positive_text = get_text(trainset, 1)

# Generate word counts for negative tone.
negative_counts = count_text(negative_text)
# Generate word counts for positive tone.
positive_counts = count_text(positive_text)

#print("Negative text sample: {0}".format(negative_text[:100]))
#print("Positive text sample: {0}".format(positive_text[:100])) ---for testing 

def get_y_count(score):
  # Compute the count of each classification occuring in the data.
  return len([r for r in trainset if r[1] == str(score)])
# We need these counts to use for smoothing when computing the prediction.
positive_review_count = get_y_count(1)
negative_review_count = get_y_count(0)

# These are the class probabilities (we saw them in the formula as P(B)).
prob_positive = positive_review_count / len(trainset)
prob_negative = negative_review_count / len(trainset)

def make_class_prediction(text, counts, class_prob, class_count):
  prediction = 1
  text_counts = Counter(re.split("\s+", text))
  for word in text_counts:
      # For every word in the text, we get the number of times that word occured in the reviews for a given class, add 1 to smooth the value, and divide by the total number of words in the class (plus the class_count to also smooth the denominator).
      # Smoothing ensures that we don't multiply the prediction by 0 if the word didn't exist in the training data.
      # smooth the denominator counts to keep things even and prevent division by zero condition
      prediction *=  text_counts.get(word) * ((counts.get(word, 0) + 1) / (sum(counts.values()) + class_count))
  # Now multiplying by the probability of the class:positive and negative, existing in the documents.
  return prediction * class_prob

print("Review: {0}".format(trainset[13][0]))
print("Negative prediction: {0}".format(make_class_prediction(trainset[13][0], negative_counts, prob_negative, negative_review_count)))
print("Positive prediction: {0}".format(make_class_prediction(trainset[13][0], positive_counts, prob_positive, positive_review_count)))



#testing algorithm on test dataset

def make_decision(text, make_class_prediction):
    # Compute the negative and positive probabilities.
    negative_prediction = make_class_prediction(text, negative_counts, prob_negative, negative_review_count)
    positive_prediction = make_class_prediction(text, positive_counts, prob_positive, positive_review_count)

    # We assign a classification based on which probability is greater.
    if negative_prediction > positive_prediction:
      return 0  #for negative predicted
    return 1  #for positive predicted

with open("test.csv", 'r') as file:
    test = list(csv.reader(file))

prediction = [make_decision(test[20][0], make_class_prediction)]
print("review test:{0}".format(test[20][0]))
print(prediction)

#error calculation through ROC curve
predictions = [make_decision(r[0], make_class_prediction) for r in test]

actual = [int(r[1]) for r in test]

from sklearn import metrics

# Generating the roc curve using scikit-learn for error computation
fpr, tpr, thresholds = metrics.roc_curve(actual, predictions, pos_label=1)

# Measuring the area under the curve.  The closer to 1, the "better" the predictions.
print("AUC of the predictions: {0}".format(metrics.auc(fpr, tpr)))
import re
import math

class Bayes_Classifier:
    def __init__(self):
        # Initialize necessary variables for the Naive Bayes Classifier
        self.log_class_priors = {} # Dictionary to store the log of the prior probabilites of each class
        self.word_counts = {} # Nested dictionary to store the counts of each word for each class
        self.vocab = set() # Set to keep track of all the unique words(vocabulary) encountered during training
        self.class_counts = {} # A dictionary to count the numbers of documents for each class
    '''
    def stem(self, word):
        # Define some simple regex patterns for stemming
        plural = re.compile('(ies|es|s)$')
        past_simple = re.compile('(ed)$')
        continuous = re.compile('(ing)$')
    
        # Remove 'ies', 'es', 's' if it's a plural form
        if plural.search(word):
            word = plural.sub('', word)
            if word[-1] == 'i':  # If we removed 'ies', replace 'i' with 'y'
                word = word[:-1] + 'y'
        # Remove 'ed' if it's a past simple form
        elif past_simple.search(word):
            word = past_simple.sub('', word)
            if word[-2:] == 'i':  # If we removed 'ed' after 'i', replace 'i' with 'y'
                word = word[:-2] + 'y'
        # Remove 'ing' if it's a continuous form and add 'e' if necessary
        elif continuous.search(word):
            word = continuous.sub('', word)
            if len(word) > 0 and word[-1] in 'aeiou' and word[-2] not in 'aeiou':
                word += 'e'
    
        return word '''

    
    def preprocess(self, text):
        # Tokenize and remove non-alphabetic characters and stop-words
        
        stop_words = set(["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", 
                          "any", "are", "as", "at", "be", "because", "been", "before", "being", "below",
                          "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", 
                          "during", "each", "few", "for", "from", "further", "had", "has", "have",
                          "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", 
                          "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", 
                          "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me", 
                          "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", 
                          "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same",
                          "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", 
                          "that", "that's", "the", "their", "theirs", "them", "themselves", "then", 
                          "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", 
                          "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", 
                          "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", 
                          "when's", "where", "where's", "which", "while", "who", "who's", "whom", 
                          "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've",
                          "your", "yours", "yourself", "yourselves"])

        tokens = re.findall(r'\b\w+\b', text.lower())
        filtered_tokens = [word for word in tokens if word  not in stop_words]
        #stemmed_tokens = [self.stem(word) for word in filtered_tokens]
        #print(filtered_tokens)
        #return stemmed_tokens

        # Generate bigrams (Accuracy actually decreased after applying biagrams)
        #bigrams = zip(filtered_tokens[:-1], filtered_tokens[1:])
        #bigram_list = ["_".join(bigram) for bigram in bigrams]

        return filtered_tokens 

    # Method to train the Naive Bayes classifier using the training data
    def train(self, data):
        # Count the frequency of each class (1 or 5 stars) and each word given a class
        for line in data:
            # Each line is of the form: 'NUMBER OF STARS|ID|TEXT'
            sentiment, _, text = line.split('|')
            sentiment = sentiment.strip()
            self.class_counts[sentiment] = self.class_counts.get(sentiment, 0) + 1
            words = self.preprocess(text)
            if sentiment not in self.word_counts:
                self.word_counts[sentiment] = {}
            for word in words:
                self.vocab.add(word)
                self.word_counts[sentiment][word] = self.word_counts[sentiment].get(word, 0) + 1

        # Calculate the log probability of each class and the log likelihood of each word given a class
        total_docs = sum(self.class_counts.values())
        self.log_class_priors = {cls: math.log(count / total_docs) for cls, count in self.class_counts.items()}

    def classify(self, data):
        results = []
        for line in data:
            _, _, text = line.split('|')
            words = self.preprocess(text)
            class_scores = {}
            for cls in self.class_counts:
                class_scores[cls] = self.log_class_priors[cls]

            # Calculate the score of each class for the document
            for word in words:
                if word not in self.vocab:
                    continue  # Ignore words not seen in training
                for cls in class_scores:
                    word_count = self.word_counts[cls].get(word, 0)
                    # Implementing add_one smoothing to handle the cases of zero probabilities
                    word_likelihood = (word_count + 1) / (sum(self.word_counts[cls].values()) + len(self.vocab))
                    class_scores[cls] += math.log(word_likelihood)

            # Predict the class with the highest score
            predicted_class = max(class_scores, key=class_scores.get)
            results.append(predicted_class)

        return results

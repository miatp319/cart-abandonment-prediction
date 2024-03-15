# cart-abandonment-prediction

The repository is part of my assignment during the course "Analysis of Customer Data". This is a simplified version of the Coveo Data Challenge organised at the eCommerce workshop co-located with the Special Interest Group for Information Retrieval (SIGIR) Conference in July 2021 in Montreal. The goal will be to perform early prediction about whether a user will abandon their cart in the current shopping session after at least a product has been added to the cart. This is therefore a binary classification problem: a session can either feature in the abandon category or in the purchase category.

In this assignment, I performed the preprocessing and perform 4-gram Naive Bayes to predict the cart abandonment 

## 1. Data overview
The training set contains 1,974,586 rows. Each row is an event, and a session can have many events. For each events, these information are provided:
- session_id_hash	
- event_type	
- product_action
- product_sku_hash
- server_timestamp_epoch_ms
- hashed_url

For the 4-gram Naive Bayes, only the sequence of events will be considered, which requires us to group events into sessions. 
The model will be evaluated based on an evaluation set given by the course coordinator. This evaluation set contains sessions, each session has 10 events after the first add-to-cart. If a session contains fewer events after the first add to cart before the session either stops or features a purchase, keep everything. 

## 2. Preprocessing
The following steps are implemented on the training set:

- filter out all sessions which never feature an add-to-cart event
- label the sessions (1: conversions; 0: cart-abandonement)
- trim the purchase sessions to the last event before the first purchase
- filter sessions that are too short (shorter than 5 events) or too long (longer than 100 events)
- symbolise the sessions (map each event to an integer based on inverse frequency, so the most frequent event maps to 1, the second most frequent to 2, and so on).

## 3. Model 
As the data is highly imbalanced, I used Complement Naive Bayes as the model for prediction. I also compared 2 oversampling methods: SMOTE and ADASYN. F1-score is improved compared to no sampling, and ADASYN is better than SMOTE in this case.

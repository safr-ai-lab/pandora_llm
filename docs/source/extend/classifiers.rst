Classifiers
===========

Features are statistics that may be multi-dimensional and/or nonlinear.
Classifiers serve as the bridge that transforms the features into a membership score.
We provide :code:`MembershipClassifier` and :code:`SupervisedMembershipClassifier` as base classes.
:code:`MembershipClassifier` contains the abstract methods :code:`preprocess_features` and :code:`predict_membership`.
:code:`SupervisedMembershipClassifier` implements a default StandardScaler-based :code:`preprocess_features` and additionally requires :code:`train_clf` as a method.
For features that are already membership scores, you don't need a supervised classifier and can just use the :code:`Threshold` class which implements the identity classifier.
But when we need to predict membership scores off of features, we will need a proper supervised classifier.

Let us walk through an example: :code:`DecisionTree`

.. code-block:: python

    class DecisionTree(SupervisedMembershipClassifier):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def train_clf(self,
            features: Float[torch.Tensor, "n"],
            labels: Int[torch.Tensor, "n"],
            max_depth: int=None,
            seed: int=229
        ) -> Float[torch.Tensor, "n"]:
            """
            Take train and validation data and train decision tree as MIA.

            Args:
                features: Features for training supervised MIA 
                labels: Labels for train data (binary, 1 is train)
                max_depth: Max depth of decision tree
                seed: Seed of decision tree
                        
            Returns:
                Train predictions
            """
            self.clf = DecisionTreeClassifier(max_depth=max_depth,random_state=seed).fit(features, labels)
            return -self.clf.predict_proba(features)[:,1]

        def predict_membership(self, features: Num[torch.Tensor, "n"]) -> Float[torch.Tensor, "n"]:
            """
            Compute the membership probability based on given preprocessed features. 

            Args:
                features: input data to compute statistic over
            Returns:
                Membership probabilities computed on the input features
            """
            if self.clf is None:
                raise Exception("Please call .train_model() to train the classifier first.")
            
            return -self.clf.predict_proba(features)[:,1]

:code:`DecisionTree` inherits from :code:`SupervisedMembershipClassifier`, and thus the only things that need to be implemented are :code:`train_clf` and :code:`predict_membership`, both of which are simple calls to the :code:`sklearn` API.

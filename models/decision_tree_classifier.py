from sklearn.tree import DecisionTreeClassifier
from models.defaults import DEFAULTS
try:
    from StringIO import StringIO ## for Python 2
except ImportError:
    from io import StringIO ## for Python 3
# try:
#     import pydotplus
# except ImportError:
#     print("installing pydotplus---->")
#     os.system("conda install pydotplus")
#     import pydotplus
#
# try:
#     import graphviz
# except ImportError:
#     print("installing graphviz---->")
#     os.system("conda install python-graphviz")
#     import graphviz

class DTClassifier():

    def __init__(self,dataset):
        self.dataset = dataset
        self.decisiontree = DecisionTreeClassifier(**DEFAULTS[dataset]['dt']['defaults'])
        print("""
    		**********************
    		Decision Tree Classifier
    		**********************
    	""")

    def train_and_predict(self, X, y, X_test):
        '''
        fit training dataset and predict values for test dataset
        '''
        self.decisiontree.fit(X, y)
        self.decisiontree.predict(X_test)

    def score(self, X, X_test, y, y_test):
        '''
        Returns the score of Decision Tree by fitting training data
        '''
        self.train_and_predict(X, y, X_test)
        return self.decisiontree.score(X_test, y_test)

    def create_new_instance(self, values):
        return DecisionTreeClassifier(**{**values})

    def param_grid(self, is_random=False):
        '''
        dictionary of hyper-parameters to get good values for each one of them
        '''
        # random search only accepts a dict for params whereas gridsearch can take either a dic or list of dict
        return DEFAULTS[self.dataset]['dt']['param_grid']

    def get_sklearn_model_class(self):
        return self.decisiontree

    def plot_and_save_tree(self, max_depth=None):
        # pass
        file_name = "./tree/{}".format(self.dataset.split('/')[-1])
        # try:
        #     with open(file_name, 'x') as file:
        #         pass
        # except FileExistsError:
        #     pass
        # dot_data = StringIO()
        # tree.export_graphviz(self.decisiontree, out_file=dot_data, max_depth=max_depth)
        # graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
        # graph.write_png('{}.png'.format(file_name))

    def __str__(self):
        return "DecisionTree"
import os.path, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

from rakg import matrix_fact, datasets

G, X = datasets.load_prot_prot(on_server=True)

MF = matrix_fact.Matrix_Factorization(k = 30, G=G, X=X)

fpr, tpr, auc = MF.train_test_normal_model(n_folds = 3, 
                alpha=0.1, n_iter=2, with_sampling = False)

print auc
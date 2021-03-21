from process import load_model, load_msr

X_tr, y_tr = load_msr('data/msr/msr_paraphrase_train.txt')
X_test, y_test = load_msr('data/msr/msr_paraphrase_test.txt')

print(X_tr[0][0])

text = ' '.join(X_tr[0][0])

print(text)

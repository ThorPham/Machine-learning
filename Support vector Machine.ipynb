{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# Support vector machine là một model rất mạnh và được sử dụng rộng rãi trong machine learning. Thuật toán này vừa áp dụng\n",
    "# được cho cả Linear và non-linear classifier,regression hay thậm chí cả detection outlier.SVM thể hiện tốt hơn đối với dữ liệu\n",
    "# small or median size."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## SVM for Linear Classifier\n",
    "Ý tưởng đằng sau SVM có thể hình dung như hình bên dưới. Hai class có thể dễ dàng phân tách với nhau bằng một đường thằng \n",
    "tuyến tính (linear separable). Vấn đề là tìm đường thằng nào cho nó tối ưu vì có rất nhiều đường thẳng có thể classifier .\n",
    "Đường thẳng tối ưu sẽ là đường mà nó có hai đường biên là lớn nhất tức là large margin classification.\n",
    "Chú ý rằng khi ta add thêm dữ liệu vào thì không ảnh hưởng đến các tham số của model đã được predict .Vì các tham số này được predict từ những point gọi là vector support tức là nó đã được chọn để optimizer margin nên nó sẽ không thay đổi.\n",
    "Một điểm nữa cần lưu ý là SVM rất nhạy đối với feature scale tức là thang đo của mỗi feature.Chúng ta thường scaling feature trước khi training model."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Soft margin classifier\n",
    "Ở trên chúng ta đang nói về SVM classication cho hard margin tức là  mỗi điểm dữ liệu sẽ thuộc về 1 class và không có instance nào ở giữa decision boundry.Tuy nhiên,trong thực tế dataset thường có rất nhiều outlier nên hard margin khó có thể giải quyết được . Do đó người ta cho phép ở giữa decission boundry có một số lượng instance nhất định (margin violation) hay còn gọi là soft margin.\n",
    "Trong sklearn người ta có thể kiểm soát violent này bằng tham số C. C nhỏ dẫn tới decision boudry lớn nên margin violation cũng lớn theo và ngược lại."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Pipeline(steps=[('Scaler', StandardScaler(copy=True, with_mean=True, with_std=True)), ('linear_svc', LinearSVC(C=1, class_weight=None, dual=True, fit_intercept=True,\n",
       "     intercept_scaling=1, loss='hinge', max_iter=1000, multi_class='ovr',\n",
       "     penalty='l2', random_state=None, tol=0.0001, verbose=0))])"
      ]
     },
     "execution_count": 51,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# SVM với dataset iris \n",
    "import numpy as np\n",
    "from sklearn.pipeline import Pipeline\n",
    "from sklearn import datasets\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.svm import LinearSVC\n",
    "# Ta chỉ dự đoán có phải Iris-Virginica dựa trên petal length, petal width\n",
    "iris = datasets.load_iris()\n",
    "X = iris[\"data\"][:, (2, 3)]  # petal length, petal width\n",
    "y = (iris[\"target\"] == 2).astype(np.float64)  # Iris-Virginica\n",
    "svm_clf = Pipeline([(\"Scaler\",StandardScaler()),(\"linear_svc\",LinearSVC(C=1, loss=\"hinge\"))])\n",
    "svm_clf.fit(X,y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ 1.])"
      ]
     },
     "execution_count": 53,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# predict\n",
    "svm_clf.predict([[4.2,2]])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "metadata": {},
   "outputs": [],
   "source": [
    "# lưu ý trong SVM output là label không phải là probabilities như logistic regression\n",
    "# Khi dữ liệu chúng ta lớn thì SVM(kernel = \"linear\",C=1) training rất chậm và không được khuyến khích.Phương pháp thay thế là\n",
    "# SGDClassifier(loss=\"hinge\",alpha =1/(m*C)) sử dung stochatics gradient descent training SVM tuy nó không nhanh hơn LinearSVC\n",
    "# nhưng khi dữ liệu lớn thì thời gian training vẫn không thay đổi nhiều\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

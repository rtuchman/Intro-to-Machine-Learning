import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn import svm
import numpy as np
plt.ioff()  # Turn interactive plotting off
import random
import warnings
warnings.filterwarnings("ignore")  # ignore Warnings: living on the edge



def plot_vector_as_image(image, h, w, title='img'):
	"""
	utility function to plot a vector as image.
	Args:
	image - vector of pixels
	h, w - dimesnions of original pi
	"""
	plt.imshow(image.reshape((h, w)), cmap=plt.cm.gray)
	plt.title(title, size=12)
	plt.show()


def get_pictures_by_name(name='Ariel Sharon'):
    """
    Given a name returns all the pictures of the person with this specific name.
    YOU CAN CHANGE THIS FUNCTION!
    THIS IS JUST AN EXAMPLE, FEEL FREE TO CHANGE IT!
    """
    lfw_people = load_data()
    selected_images = []
    n_samples, h, w = lfw_people.images.shape
    target_label = list(lfw_people.target_names).index(name)
    for image, target in zip(lfw_people.images, lfw_people.target):
        if (target == target_label):
            image_vector = image.reshape((h*w, 1))
            selected_images.append(image_vector)
    return selected_images, h, w

def load_data():
    # Don't change the resize factor!!!
    lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
    return lfw_people

######################################################################################
"""
Other then the PCA function below the rest of the functions are yours to change.
"""

def PCA(X, k):
    """
    Compute PCA on the given matrix.

    Args:
        X - Matrix of dimesions (n,d). Where n is the number of sample points and d is the dimension of each sample.
        For example, if we have 10 pictures and each picture is a vector of 100 pixels then the dimesion of the matrix would be (10,100).
        k - number of eigenvectors to return

    Returns:
      U - Matrix with dimension (k,d). The matrix should be composed out of k eigenvectors corresponding to the largest k eigenvectors
            of the covariance matrix.
      S - k largest eigenvalues of the covariance matrix. vector of dimension (k, 1)
    """
    u = X.mean(axis=0)
    X = X - u
    cov = np.matmul(np.transpose(X), X)/len(X)
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    U = np.transpose(eigenvectors[:, :k])
    S = np.array([[val] for val in eigenvalues[:k]])
    return U, S

def qa_ii():
    selected_images, h, w = get_pictures_by_name('Donald Rumsfeld')
    reshaped_images = [x.reshape(1, h*w)[0, :] for x in selected_images]
    reshaped_images = np.array(reshaped_images)
    U, S = PCA(reshaped_images, 10)
    plot_all_vectors(U, h, w, 'Donald Rumsfeld - PCA')

def qa_iii():
    selected_images, h, w = get_pictures_by_name('Donald Rumsfeld')
    reshaped_images = [x.reshape(1, h * w)[0, :] for x in selected_images]
    reshaped_images = np.array(reshaped_images)

    distances = []
    k_list = [1, 5, 10, 30, 50, 100]
    samples = random.sample(range(len(reshaped_images)), 5)
    for k in k_list:
        U, S = PCA(reshaped_images, k)
        l2_sum = 0
        for s in samples:
            x = reshaped_images[s, :]
            a = encode(U, x)
            x_transformed = decode(U, a)
            fig, (ax1, ax2) = plt.subplots(1, 2)
            ax1.axis('off')
            ax1.imshow(x.reshape((h, w)), cmap=plt.cm.gray)
            ax1.set_title('original img{}'.format(s))
            ax2.axis('off')
            ax2.imshow(x_transformed.reshape((h, w)), cmap=plt.cm.gray)
            ax2.set_title('transformed img{}'.format(s))
            fig.suptitle('Original and transformed for k={}'.format(k))
            fig.savefig(r'plots/qa_3_img{}'.format(s))
            plt.close()
            l2_sum += np.linalg.norm(x-x_transformed)
        distances.append(l2_sum)

    plt.plot(k_list, distances)
    plt.xlabel('k')
    plt.ylabel('L2 dist')
    plt.title('Sum of L2 distances as function of k')
    plt.savefig(r'plots/qa_3_l2_distances.png')
    plt.close()

def q3_iv():
    names = load_data()['target_names']
    first = True
    for n in names:
        ims, h, w = get_pictures_by_name(n)
        reshaped_images = [x.reshape(1, h * w)[0, :] for x in ims]
        reshaped_images = np.array(reshaped_images)

        if len(ims[0]) >= 70:
            if first:
                X = reshaped_images
                y = np.array([n]*len(ims))
                first = False
            else:
                X = np.concatenate((X, reshaped_images))
                tmp_y = np.array([n]*len(ims))
                y = np.concatenate((y, tmp_y))



    parameters = {'kernel': ['rbf'], 'C': [10, 100, 1000, 10000], 'gamma': [1e-7, 1e-8, 1e-9, 1e-10]}
    k_list = [1, 5, 10, 30, 50, 100, 150, 300]
    accuracy_list = []
    for k in k_list:
        U, S = PCA(X, k)
        encoded_list = np.array([encode(U, x) for x in X])
        X_train, X_test, y_train, y_test = train_test_split(encoded_list, y, test_size=0.25, random_state=22)
        svc = svm.SVC()
        clf = GridSearchCV(svc, parameters, scoring='accuracy',)
        clf.fit(X_train, y_train)
        accuracy_list.append(sum(clf.predict(X_test) == y_test)/ len(X_test))

    max_acc = max(accuracy_list)
    plt.title('Accuracy as function of k (max={})'.format(max_acc))
    plt.plot(k_list, accuracy_list)
    plt.grid()
    plt.xlabel('k')
    plt.ylabel('Accuracy')
    plt.savefig(r'plots/qa_iv2.png')
    plt.close()

def encode(U, x):
    return np.matmul(U, np.transpose(x))

def decode(U, a):
    return np.matmul(np.transpose(U), a)



def plot_all_vectors(images, h, w, title='img'):
    """saves all vectors in images as individual pictures and also in subplots"""
    for j in range(len(images)):
        plt.imshow(images[j, :].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(r'plots/{}_v{}'.format(title, j))
        #plt.show()
        plt.savefig(r'plots/{}_v{}'.format(title, j))
        plt.close()

    fig, ax = plt.subplots(2, 5)
    i = 0
    for row in ax:
        for col in row:
            col.axis('off')
            col.imshow(images[i, :].reshape((h, w)), cmap=plt.cm.gray)
            i += 1
    plt.savefig(r'plots/{} all'.format(title))


if __name__ == "__main__":
    qa_ii()
    qa_iii()
    q3_iv()

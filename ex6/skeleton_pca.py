import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
import numpy as np
plt.ioff()  # Turn interactive plotting off
from numpy.linalg import svd

def plot_vector_as_image(images, h, w, title='img'):
    """
    utility function to plot a vector as image.
    Args:
    image - vector of pixels
    h, w - dimesnions of original pi
    """
    fig, ax = plt.subplots(2, 5)

    i = 0
    for row in ax:
        for col in row:
            col.axis('off')
            col.imshow(images[i, :].reshape((h, w)), cmap=plt.cm.gray)
            i += 1

    plt.savefig(title)

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
    plot_vector_as_image(U, h, w, 'Donald Rumsfeld - PCA')

def qa_iii():
    pass

if __name__ == "__main__":
    qa_ii()
    #a = load_data()
    #data = a['images'][:10]
    #data = [x.reshape(1, 1850)[0, :] for x in data]
    #data = np.array(data)
    #PCA(data, 5)

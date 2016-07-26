# Author: Almero Gouws <14366037@sun.ac.za>
"""
This file can be used to denoise single channel images using approximate
MAX-SUM inference on a Markov Random Field.
"""
import numpy as np
from scipy import sparse
from scipy.misc.pilutil import imread, imsave
import pylab
import models
import cliques
import inference
import potentials
from general import subv2ind
import time
import psyco
psyco.full()


def prompt(prompt):
    """
    Displays a message in the console, prompting the user for input, then
    captures a keyboard string input from the user. The user terminates
    input by pressing the 'Enter' key. The function then returns the
    captured string.

    Parameters
    ----------
    prompt: String
        A message to prompt the user what to enter. (String)
    """
    return raw_input(prompt).strip()

def set_adj_mat_entry(adj_mat, ind):
    """
    """
    if adj_mat[ind[1], ind[0]] != 1:
        adj_mat[ind[0], ind[1]] = 1

    return adj_mat
        

def create_lattice_sparse_adj_matrix(rows, cols, layers=1):
    """
    Creates an adjacency matrix for a lattice shaped graph with an
    arbitrary number of rows and columns.

    NOTE:
    The subv2ind routine used in the function works in column-major order,
    and this function assumes row-major order, therefore row and column
    indices have been switched round when using the subv2ind function.
    """
    """Create adjacency matrix"""
    adj_mat = sparse.lil_matrix((rows*cols*layers, rows*cols*layers), dtype=int)
    
    """Assign the 2 edges for top-left node"""
    adj_mat[0, 1] = 1
    adj_mat[0, cols] = 1

    """Assign the 2 edges for top-right node"""
    ind = subv2ind(np.array([cols, rows]), np.array([cols-1, 0]))[0, 0]
    temp_ind = subv2ind(np.array([cols, rows]), np.array([cols-2, 0]))[0, 0]
    adj_mat = set_adj_mat_entry(adj_mat, [ind, temp_ind])
    temp_ind = subv2ind(np.array([cols, rows]), np.array([cols-1, 1]))[0, 0]
    adj_mat = set_adj_mat_entry(adj_mat, [ind, temp_ind])

    """Assign the 2 edges for bottom-left node"""
    ind = subv2ind(np.array([cols, rows]), np.array([0, rows-1]))[0, 0]
    temp_ind = subv2ind(np.array([cols, rows]), np.array([0, rows-2]))[0, 0]
    adj_mat = set_adj_mat_entry(adj_mat, [ind, temp_ind])
    temp_ind = subv2ind(np.array([cols, rows]), np.array([1, rows-1]))[0, 0]
    adj_mat = set_adj_mat_entry(adj_mat, [ind, temp_ind])

    """Assign the 2 edges for bottom_right node"""
    ind = subv2ind(np.array([cols, rows]), np.array([cols-1, rows-1]))[0, 0]
    temp_ind = subv2ind(np.array([cols, rows]), np.array([cols-2, rows-1]))[0, 0]
    adj_mat = set_adj_mat_entry(adj_mat, [ind, temp_ind])
    temp_ind = subv2ind(np.array([cols, rows]), np.array([cols-1, rows-2]))[0, 0]
    adj_mat = set_adj_mat_entry(adj_mat, [ind, temp_ind])

    """Assign the 3 edges for each left border nodes"""
    for i in range(1, rows-1):
        ind = subv2ind(np.array([cols, rows]), np.array([0, i]))[0, 0]
        temp_ind = subv2ind(np.array([cols, rows]), np.array([0, i-1]))[0, 0]
        adj_mat = set_adj_mat_entry(adj_mat, [ind, temp_ind])
        temp_ind = subv2ind(np.array([cols, rows]), np.array([0, i+1]))[0, 0]
        adj_mat = set_adj_mat_entry(adj_mat, [ind, temp_ind])
        adj_mat[ind, ind+1] = 1

    """Assign the 3 edges for each right border nodes"""
    for i in range(1, rows-1):
        ind = subv2ind(np.array([cols, rows]), np.array([cols-1, i]))[0, 0]
        temp_ind = subv2ind(np.array([cols, rows]), np.array([cols-1, \
                                                              i-1]))[0, 0]
        adj_mat = set_adj_mat_entry(adj_mat, [ind, temp_ind])
        temp_ind = subv2ind(np.array([cols, rows]), np.array([cols-1, \
                                                              i+1]))[0, 0]
        adj_mat = set_adj_mat_entry(adj_mat, [ind, temp_ind])
        adj_mat = set_adj_mat_entry(adj_mat, [ind, ind-1])
        
    """Assign the 3 edges for each top border nodes"""
    for i in range(1, cols-1):
        ind = subv2ind(np.array([cols, rows]), np.array([i, 0]))[0, 0]
        adj_mat = set_adj_mat_entry(adj_mat, [ind, ind-1])
        adj_mat = set_adj_mat_entry(adj_mat, [ind, ind+1])
        temp_ind = subv2ind(np.array([cols, rows]), np.array([i, 1]))[0, 0]
        adj_mat = set_adj_mat_entry(adj_mat, [ind, temp_ind])

    """Assign the 3 edges for each bottom border nodes"""
    for i in range(1, cols-1):
        ind = subv2ind(np.array([cols, rows]), np.array([i, rows-1]))[0, 0]
        adj_mat = set_adj_mat_entry(adj_mat, [ind, ind-1])
        adj_mat = set_adj_mat_entry(adj_mat, [ind, ind+1])
        temp_ind = subv2ind(np.array([cols, rows]), np.array([i, rows-2]))[0, 0]
        adj_mat = set_adj_mat_entry(adj_mat, [ind, temp_ind])

    """Assign edges for inner, fully-connected nodes"""
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            ind = subv2ind(np.array([cols, rows]), np.array([j, i]))[0, 0]
            temp_ind = subv2ind(np.array([cols, rows]), np.array([j, i-1]))[0, 0]
            adj_mat = set_adj_mat_entry(adj_mat, [ind, temp_ind])
            adj_mat = set_adj_mat_entry(adj_mat, [ind, ind-1])
            adj_mat = set_adj_mat_entry(adj_mat, [ind, ind+1])
            temp_ind = subv2ind(np.array([cols, rows]), np.array([j, i+1]))[0, 0]
            adj_mat = set_adj_mat_entry(adj_mat, [ind, temp_ind])

    """Assign the edges to the observed nodes"""
    for i in range(0, adj_mat.shape[0]/2):
        adj_mat[i, i + adj_mat.shape[0]/2] = 1      
    
    return adj_mat
    

def create_img_prob_table(depth, sigma=0.5):
    """
    Creates a lookup table of probabilities for a 2 node clique in the
    ising model used for graphical model image denoising.

    Parameters
    ----------
    depth: Int
        The number of different values a pixel can take.

    sigma: Float
        Co-efficient used in creating the table.
    """
    """Initialize the table to the right size"""
    prob_tbl = np.zeros((depth, depth))

    """
    Fill the table with values, where the probability of the two nodes in
    the clique having the same value is the highest, and the probability
    of the nodes having completley opposite values is the lowest.
    """
    for i in range(0, depth):
        for j in range(0, depth):
            prob_tbl[i, j] = (-1*(i - j)**2)/2*sigma

    return prob_tbl

def denoise_image(img, depth=255, max_iter=10):
    """
    Denoises a single channel image.

    Parameters
    ----------
    img: Numpy array
        The image to denoise.

    depth: Int
        The number of different values a pixel can take.

    max_iter: Int
        The maximum number of times the inference algorithm can iterate.
    """
    """
    Create adjacency matrix representation of the hidden lattice graph that
    represents the denoised image.
    """
    print "Creating initial adjacency matrix..."
    adj_mat = create_lattice_sparse_adj_matrix(img.shape[0], img.shape[1], 2)

    print "Determine the cliques from the adjacency matrix..."   
    """Get the cliques as a list"""
    clq_doms = []
    i = 0
    for cols in adj_mat.rows:
        if len(cols) > 0:
            for col in cols:
                new_clique = [i, col]
                new_clique.sort()
                clq_doms.append(new_clique)
        i = i + 1

    """Create list of node sizes"""
    print "Creating list of node sizes..."
    ns = depth * np.ones((1, img.shape[0]*img.shape[1]*2))

    """Create list of cliques and assign potentials to them"""
    print "Creating the  list of cliques and their potentials..."
    """Image model"""
    T_img = create_img_prob_table(depth, 1)
    
    """Noise model"""
    T_noise = create_img_prob_table(depth, 5)

    clqs = []
    outer_layer = range(img.shape[0]*img.shape[1], img.shape[0]*img.shape[1]*2)
    for i in range(0, len(clq_doms)):
        if clq_doms[i][1] in outer_layer:
            clqs.append(cliques.discrete_clique(i, clq_doms[i], \
                                                np.array([depth, depth]),\
                                                T_img))
        else:
            clqs.append(cliques.discrete_clique(i, clq_doms[i], \
                                                np.array([depth, depth]), \
                                                T_noise))
            
    """Create the MRF object and set the lattice flag to TRUE"""
    print "Creating MRF..."
    net = models.mrf(adj_mat, ns, clqs, lattice=True)

    """Initialize the inference engine to be approximate"""
    net.init_inference_engine(exact=False, max_iter=max_iter)
   

    """Create the evidence, with the noisy nodes being observed"""
    evidence = img.flatten().tolist()
    N = len(evidence)
    for i in range(0, N):
        evidence.insert(0, [])

    """Run loopy-belief propagation"""
    print "Running loopy belief propagation..."    
    mlc = net.max_sum(evidence)

    """
    Extract denoised image from most likely configuaration of the hidden
    nodes.
    """
    print "Extracting denoised image..."
    new_img = np.array(mlc[0:img.shape[0]*img.shape[1]])
    new_img = np.array(new_img.reshape(img.shape[0], img.shape[1]))

    """Delete objects"""
    del img
    del adj_mat
    del net

    return new_img

if __name__ == '__main__':   
    """Define the noisy image location"""
    in_fname = 'noisy.png'

    """Define the output path to save intermediate denoised images to"""
    out_fname = '.\\output\\'

    """Load the image"""
    img = np.array(imread(in_fname, 1)/255,  dtype=int)

    """Determine the images depth"""
    depth = np.max(img)+1

    """Define the sliding window size"""
    seg_size = 100

    """
    If the image is smaller that the sliding window, then just denoise the
    whole image
    """
    if img.shape[0]<seg_size:
        new_img = denoise_image(img)
        imsave(out_fname + str(1) + '.png', new_img)
    else:
        """Denoise the image in overlapping segments"""
        count = 0
        cut = int(float(seg_size)/2)
        for i in range(cut, img.shape[0], cut):
            for j in range(cut, img.shape[1], cut):
                """Extract the window to denoise"""
                sub_img = img[i-cut:i+cut, j-cut:j+cut]

                """Denoise the window"""
                new_img = denoise_image(sub_img, depth, 6)

                """
                Place the denoised window back into the noisy image, except
                for the leading edge pixels of the window.
                """
                img[(i-cut):(i+cut-1), (j-cut):(j+cut-1)] = \
                                   new_img[0:seg_size-1, 0:seg_size-1]

                """Compensate for edge cases"""
                if (i + cut) == img.shape[0]:
                    img[i+cut-1, j-cut:j+cut] = \
                                   new_img[seg_size-1, 0:seg_size]
                if (j + cut) == img.shape[1]:
                    img[i-cut:i+cut, j+cut-1] = \
                                   new_img[0:seg_size, seg_size-1]

                print "Saving partially denoised image..."
                imsave(out_fname + str(count+1) + '.png', img)
                count = count + 1

    prompt("Press enter to exit...")

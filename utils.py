import torch
import torch.utils.data
from torchvision import datasets, transforms
import torchvision.datasets as dataset
from random import randint
import numpy
import matplotlib
# @yuchen
matplotlib.use("AGG")
import matplotlib.pyplot as plt
import pickle
from torch.autograd import grad, Variable
from os import path, makedirs
import torch.nn.functional as F
from tqdm import tqdm

def resize_image(mnist_images):
    # We wish to resize the mnist images from 28x28 to 32x32
    n, w, h = mnist_images.size()
    data = torch.zeros(n, 32, 32)
    data[:,2:30,2:30] = mnist_images
    return data

def get_data_synthetic():
    '''
    
    Generates random, uniform 2D Gaussians centered at:
    
    +-----+-----+-----+
    |     |     |     |
    |  1  |  2  |  3  |
    |     |     |     |
    +-----+-----+-----+
    |     |     |     |
    |  4  |  5  |  6  |
    |     |     |     |
    +-----+-----+-----+
    |     |     |     |
    |  7  |  8  |  9  |
    |     |     |     |
    +-----+-----+-----+
    
    Each image is (32, 32), with means of Gaussians at 
    5, 15 and 25 of the intersections of both axis.
    
    '''
    TOTAL = 20000
    DENSITY = 2000 # number of points to sample
    
    NUM_CLASSES = 9
    CHANNELS = 1
    
    means = [(j, i) for i in range(5, 26, 10) for j in range(5, 26, 10)]
    covar = [[2, 0], [0, 2]] # 1 -> radius of 3, 2 -> radius of 5
    
    train_data = [None]*TOTAL
    train_labels = [None]*TOTAL
    
    outdir = "synthetic"
    if not path.isdir(outdir):
        makedirs(outdir)
    
    # Note: pointer assignments are faster in Python than Numpy/Pytorch through Python.
    for i in tqdm(range(TOTAL), desc="Generating Gaussians", ncols=80):
        label = randint(0, NUM_CLASSES-1)
        train_labels[i] = label
        mean = means[label]
        data = numpy.zeros((32, 32))
        x, y = numpy.random.multivariate_normal(mean, covar, DENSITY).T
        keep = (x >= 0) & (x <= 31) & (y >= 0) & (y <= 31)
        x = numpy.rint(x)[keep].astype(numpy.int32)
        y = numpy.rint(y)[keep].astype(numpy.int32)
        data[x, y] = 1.0
        train_data[i] = torch.from_numpy(data).float()
        
        #save_image(data, "{:0>6}".format(i), outdir, CHANNELS)
        
    train_data = torch.cat(train_data).view(TOTAL, CHANNELS, 32, 32)
    train_labels = onehot(torch.LongTensor(train_labels), NUM_CLASSES).view(TOTAL, 1, 1, NUM_CLASSES)
    
    test_data = torch.FloatTensor([])
    test_labels = torch.FloatTensor([])
    
    return train_data, train_labels

def onehot(indices, d):
    base = torch.zeros((indices.size(0), d))
    return base.scatter_(1, indices.view(-1, 1), 1)

def get_data():
    CIFAR10_train = dataset.CIFAR10(root="./data", train=True, transform=transforms.Compose(
        [transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]), download=True)
    CIFAR10_train_data = [data[0].numpy() for data in CIFAR10_train]
    
    CIFAR10_train_data = torch.FloatTensor(CIFAR10_train_data)
    CIFAR10_train_labels = torch.LongTensor(CIFAR10_train.train_labels)
    return CIFAR10_train_data, CIFAR10_train_labels

def create_dataset(mnist_train_data, mnist_train_labels, batch_size):
    n = len(mnist_train_labels)
    # one hot encoding
    labels = torch.zeros(n, 10).scatter_(1, mnist_train_labels.view(n, 1), 1)
    dataset = torch.utils.data.TensorDataset(mnist_train_data, labels)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory = True)

def create_dataset_yuchen(data, labels, batch_size):
    dataset = torch.utils.data.TensorDataset(data, labels)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)

def combine_to_tensor(gen_list):
    return torch.cat(gen_list, dim=0)

def exp_decay_loss(scores, constant, batchSize, windowSize, gpu_mode):
    n, d = scores.size()
    assert d == 1
    assert n % batchSize == 0
    numEditors = n // batchSize
    chunks = scores.view(numEditors, batchSize)
    factor = -constant/float(windowSize)
    decay = torch.exp(torch.arange(numEditors)*factor)
    
    if gpu_mode:
        decay = decay.cuda()
    
    means = chunks.mean(dim=1)
    assert means.size() == decay.size()
    
    return (means*decay).sum()

#def grad_penalty(real_data, fake_data, penalty, discriminator, labels = None):
#    alpha = torch.cuda.FloatTensor(fake_data.shape[0], 1, 1, 1).uniform_(0, 1).expand(fake_data.shape)
#    interpolates = alpha * fake_data + (1 - alpha) * real_data
#    #interpolates = Variable(interpolates, requires_grad=True)
#    disc_interpolates = discriminator(interpolates, labels)

#    grad_out = torch.ones(disc_interpolates.size()).cuda()
#        
#    gradients = grad(
#        outputs=disc_interpolates,
#        inputs=interpolates,
#        grad_outputs = grad_out
#    )[0]
#                              #create_graph=True, retain_graph=True, only_inputs=True)[0]
#    gradients = gradients.view(gradients.size(0), -1)
#    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
#    return penalty * gradient_penalty

# @yuchen unoptimized code below

def grad_penalty(real_data, fake_data, penalty, discriminator, labels = None):
    alpha = torch.cuda.FloatTensor(fake_data.shape[0], 1, 1, 1).uniform_(0, 1).expand(fake_data.shape)
    interpolates = alpha * fake_data + (1 - alpha) * real_data
    interpolates = Variable(interpolates, requires_grad=True)
    disc_interpolates = discriminator(interpolates, labels)

    grad_out = torch.ones(disc_interpolates.size()).cuda()
    gradients = grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs = grad_out,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return penalty * gradient_penalty


def create_noisy_dataset(mnist_train_data, mnist_train_labels):
    noisy_to_real = {}
    def add_noise_to_image(image, prob):
        output = numpy.zeros(image.shape)
        thres = 1 - prob 
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                rdn = numpy.random.uniform(0,1)
                output[i][j] = rdn
                if rdn > prob:
                    output[i][j] = image[i][j]
        return output

    noisy_dataset = []
    for i, image in enumerate(mnist_train_data):
        noisy_image = add_noise_to_image(image, 0.5)
        combined = combine_image_and_onehot_label(noisy_image, mnist_train_labels[i])
        noisy_dataset.append(combined)
        noisy_to_real[noisy_image.tostring()] = image
        print(i)

    noisy_dataset = torch.tensor(numpy.array(noisy_dataset))
    my_dataset = torch.utils.data.DataLoader(noisy_dataset, batch_size=100, shuffle=True, drop_last=True)
    pickle.dump(my_dataset, open("noisy_data", "wb"))
    pickle.dump(noisy_to_real, open("noisy_to_real", "wb"))
    return noisy_dataset

def show_image(data, label):
    # Plot
    data = data.permute(1,2,0)
    print(data.shape)
    plt.title("Label is " + str(label)) 
    plt.imshow(data)
    plt.show()

def save_image(data, label, directory):
    # Plot
    data = data.transpose((1,2,0))
    filename = directory + "/" + label + ".png"
    matplotlib.image.imsave(filename, data)    

def combine_image_and_onehot_label(data, label):
    id_matrix = numpy.identity(10)
    result = numpy.zeros((33, 32))
    label_result = numpy.zeros(32)
    
    label_onehot = id_matrix[label]
    label_result[0:10] = label_onehot
    
    result[:-1,:] = data
    result[-1,:] = label_result
    
    return result
    
def extract_label_and_image(combined):
    images = numpy.zeros((len(combined), 32, 32))
    labels = numpy.zeros((len(combined), 10))
    
    for i, element in enumerate(combined):
        image = element[:-1,:]
        images[i] = image
        
        onehot = element[-1,:10]
        labels[i] = onehot
        
    return torch.from_numpy(images).float(), torch.from_numpy(labels).float() 

def normal_init(m, mean, std):
    if isinstance(m, torch.nn.ConvTranspose2d) or isinstance(m, torch.nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

def histo(network, name, epoch, directory):
    params = torch.FloatTensor([]).cuda()
    for p in network.parameters():
        p1 = p.view(-1)
        params = torch.cat((params, p1), 0)
    params = params.detach().cpu().numpy()
    plt.hist(params, bins = 1000)
    plt.title('Histogram of ' + name + ' network, ep: ' + str(epoch))
    label = name + '_hist_ep_' + str(epoch)
    plt.savefig(directory + "/" + label + ".png")

def plot_editor_scores(generator, discriminator, gpu_mode, num_edits, folder, epoch, wlabels=False):
    batch = 64
    z = torch.randn((batch, 128))
    if gpu_mode:
        z = z.cuda().float()
    if wlabels:
        labels = numpy.random.choice(10,100)
        onehot_labels = numpy.eye(10)[labels]
        onehot_labels = torch.from_numpy(onehot_labels)
        onehot_labels = onehot_labels.view(-1,1,1,10).float()
        onehot_labels = onehot_labels.cuda() 
    
    if wlabels:
        gen_images = generator(z, onehot_labels)
    else:
        gen_images = generator.forwardGenerateDetached(z, 1+num_edits)
    
    G_tensor = gen_images
    editor_scores = [discriminator(G_tensor).mean().item()]
    for i in range(num_edits):
        editor_scores.append(discriminator(generator.forwardGenerateDetached(z, 1+i)).mean().item())
    edit_id = numpy.arange(0, len(editor_scores), 1)
    
    fig, ax = plt.subplots()
    ax.plot(edit_id, editor_scores)
    ax.set(xlabel='Editor #', ylabel='D scores',
                   title='Wish to see concave upward')
    ax.grid()
    plt.savefig(folder + "/" + "Epoch" + str(epoch) + ".png")
    fig.clear()
    
    print("Edit losses:" + str(editor_scores))

def compute_wass_distance(images, discriminator, generator, folder, epoch, num_editors):
    num_samples = 100
    z = torch.randn((num_samples, 128)).cuda()
    real = torch.mean(discriminator(images))
  
    generated = generator.forwardGenerate(z, num_editors+1)
    wass_distance = []
    for edit_images in generated:
        edit_images = edit_images.detach()
        edit_mean = torch.mean(discriminator(edit_images))
        wass_distance.append(real - edit_mean)

    edit_id = numpy.arange(0, len(wass_distance), 1)
    fig, ax = plt.subplots()
    ax.plot(edit_id, wass_distance)
    ax.set(xlabel='Editor #', ylabel='Wass distance',
                   title='Wish to see convex downward')
    ax.grid()
    plt.savefig(folder + "/" + "Epoch" + str(epoch) + ".png")
    fig.clear()

def wrong_label_gen(label, wrong_num):
    if label.is_cuda:
        ones = torch.ones(label.size()).cuda()
        generated_sample = torch.zeros(label.size()).cuda()
        generated_sample = torch.FloatTensor([]).cuda()
    else:
        ones = torch.ones(label.size())
        generated_sample = torch.zeros(label.size())
        generated_sample = torch.FloatTensor([])
    
    dim = label.shape[3]
    wrong_labels = ones - label
    
    for j in range(wrong_labels.shape[0]):
        wrongs = []
        for i in range(wrong_labels.shape[3]):
            if wrong_labels[j, 0, 0, i].item() != 0:
                wrongs.append(i)
        shuffle(wrongs)
        t = 0
        while t < wrong_num:
            onehot_labels = numpy.eye(dim)[wrongs[t]]
            onehot_labels = torch.from_numpy(onehot_labels)
            onehot_labels = onehot_labels.view(-1,1,1,dim).float()
            if label.is_cuda:
                onehot_labels = onehot_labels.cuda()
            generated_sample = torch.cat((generated_sample, onehot_labels), 0)
            t += 1
                
    return generated_sample

def get_var_of_data(data):
    std = torch.std(data, dim=0)
    return std 


iris = sns.load_dataset("iris")
iris = sns.load_dataset("iris")
setosa = iris.loc[iris.species == "setosa"]
virginica = iris.loc[iris.species == "virginica"]
versicolor = iris.loc[iris.species == "versicolor"]


with sns.axes_style('white'):
    x = ((setosa.sepal_width - setosa.sepal_width.min())/
         (setosa.sepal_width - setosa.sepal_width.min()).max()) - .14
    y = ((setosa.sepal_length - setosa.sepal_length.min())/
         (setosa.sepal_length - setosa.sepal_length.min()).max()) - .12
    fig = plt.figure(figsize=(2,2));
    sns.kdeplot(x, y, shade=True, n_levels=10,shade_lowest=False,
                     clip=[[-3, 3], [-3, 3]],
                     cmap = 'binary')
    plt.scatter([0.43], [0.36], s=50, edgecolors='r', facecolors='none')

    plt.xticks([])
    plt.yticks([])
    
    plt.xlabel('alpha')
    plt.ylabel('beta')
    plt.tight_layout()
plt.savefig('dist1.pdf')
with sns.axes_style('white'):
    x = ((virginica.sepal_width - virginica.sepal_width.min())/
         (virginica.sepal_width - virginica.sepal_width.min()).max())
    y = ((virginica.sepal_length - virginica.sepal_length.min())/
         (virginica.sepal_length - virginica.sepal_length.min()).max()) + .1
    fig = plt.figure(figsize=(2,2));
    sns.kdeplot(x, y, shade=True, n_levels=10, shade_lowest=False,
                     clip=[[-3, 3], [-3, 3]],
                     cmap = 'binary')
    plt.scatter([0.43], [0.36], s=50, edgecolors='r', facecolors='none')

    plt.xticks([])
    plt.yticks([])
    plt.xlim(-.25, 1.25)
    plt.ylim(-.25, 1.25)
    
    plt.xlabel('alpha')
    plt.ylabel('beta')
    plt.tight_layout()
plt.savefig('dist2.pdf')
with sns.axes_style('white'):
    x = ((versicolor.sepal_width - versicolor.sepal_width.min())/
         (versicolor.sepal_width - versicolor.sepal_width.min()).max()) 
    y = ((versicolor.sepal_length - versicolor.sepal_length.min())/
         (versicolor.sepal_length - versicolor.sepal_length.min()).max())
    fig = plt.figure(figsize=(2,2));
    sns.kdeplot(x, y, shade=True, n_levels=10, shade_lowest=False,
                     clip=[[-3, 3], [-3, 3]],
                     cmap = 'binary')
    plt.scatter([0.43], [0.36], s=50, edgecolors='r', facecolors='none')
    plt.xticks([])
    plt.yticks([])
    plt.xlim(-.25, 1.25)
    plt.ylim(-.25, 1.25)
    
    plt.xlabel('alpha')
    plt.ylabel('beta')
    plt.tight_layout()
plt.savefig('dist3.pdf')

sns.kdeplot(setosa.sepal_width, setosa.sepal_length,
                  cmap="Reds", shade=True, shade_lowest=False)
ax = sns.kdeplot(virginica.sepal_width, virginica.sepal_length,
                 cmap="Blues", shade=True, shade_lowest=False)


mean, cov = [0, 0], [(1, .5), (.25, 1)]
x, y = np.random.multivariate_normal(mean, cov, size=50).T

with sns.axes_style('white'):
    fig = plt.figure(figsize=(2,2));
    sns.kdeplot(x, y, shade=True, n_levels=50,
                     clip=[[-3, 3], [-3, 3]],
                     cmap = 'binary')
    plt.xticks([])
    plt.yticks([])
    
    plt.xlabel('alpha')
    plt.ylabel('beta')
    plt.tight_layout()

mean, cov = [.75, .5], [(1, .5), (.25, 1)]
x, y = np.random.multivariate_normal(mean, cov, size=50).T  
with sns.axes_style('white'):
    fig = plt.figure(figsize=(2,2));
    sns.kdeplot(x, y, shade=True, n_levels=50,
                     clip=[[-3, 3], [-3, 3]],
                     cmap = 'binary')
    plt.xticks([])
    plt.yticks([])
    
    plt.xlabel('alpha')
    plt.ylabel('beta')
    plt.tight_layout()

mean, cov = [.5, .25], [(1, .5), (.25, 1)]
x, y = np.random.multivariate_normal(mean, cov, size=50).T  
with sns.axes_style('white'):
    fig = plt.figure(figsize=(2,2));
    sns.kdeplot(x, y, shade=True, n_levels=50,
                     clip=[[-3, 3], [-3, 3]],
                     cmap = 'binary')
    plt.xticks([])
    plt.yticks([])
    
    plt.xlabel('alpha')
    plt.ylabel('beta')
    plt.tight_layout()
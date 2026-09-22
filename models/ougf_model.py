import numpy as np

def single_increment(Z0, t, m, theta, sigma):
    """
    Z0[:, 0] = size X
    Z0[:, 1] = OU growth rate R

    Evolves
        dX_t = R_t X_t dt
        dR_t = theta * (m - R_t) dt + sigma dB_t
    exactly over a time interval t.
    """
    # Definition of relevant constants 
    e = np.exp(- theta* t)
    
    # Constant interecept
    intercept = np.array(( 1 - e, t  + (e - 1)/theta)) * m
    
    # Linear update
    lin = np.array( ((e , 0 ), ((1 - e)/theta, 1)) )
    #Noise, Cholesky matrix
    if theta *t < 1e-3: # First order approximation, avoids negative values because of floating point
        a =  t
        b = 1 /2  * t **2
        c = 1 /3 * t**3
    else:
        a = 1/(2*theta) * (1 -e**2)
        b = 1 /(2 * theta**2) * (1 - e)**2
        c = 1 /theta**2 * (t -2/theta*(1 -e) + 1/(2*theta)*(1-e**2))
    
    v11 = np.sqrt(a)
    v21 = b/v11
    v22 = np.sqrt(c - v21**2)
    Cholesky = sigma* np.array(( (v11,0),(v21,v22)))
    
    
    # Sampling
    n_samples,_ = Z0.shape 
    G = np.random.randn(n_samples,2)
    dBt = G.dot(Cholesky.T)

    return intercept + Z0.dot(lin.T) + dBt



class Branching_OUGF():
    """ Object model of growth fragmentation with exponential growth
    Allows to simulate single samples of this model"""
    def __init__(self, ou_growth_rate_parameters ,division_rate, division_kernel):
        """
        Arguments:

        growth_rate : positive, float; growth_rate in the exponential growth
        division_rate : function, division_rate(x) is the rate of division of a polymer with trait x
        division_kernel : function, division_kernel(x) returns a random variable between (0,x) 

        """

        self.ou_growth_rate_parameters = ou_growth_rate_parameters
        self.div_rate = division_rate
        self.div_kernel = division_kernel

    def run(self, Tmax, init, max_number_of_cells = 1e4, dtmin = 1e-12, dtmax = 1e-2, dtobs = 1e-2, eps = 0.01):
        m, theta, sigma = self.ou_growth_rate_parameters
        t = 0
        Zt = np.array((init[0], np.log(init[1]))).reshape(1,2)
        Ht = np.zeros([1])
        H  = np.random.exponential(1,[1])

        snapshots = [Zt]
        times = [t]
        Nt = len(Zt)
        time_since_last_obs = 0
        while t < Tmax and Nt < max_number_of_cells:
            indices_cell_divided = np.where(Ht > H)[0]
            number_of_divisions = len(indices_cell_divided)
            old_rate = self.div_rate(Zt[:, 0], np.exp(Zt[:, 1]))
            while number_of_divisions == 0:
                max_old_rate = np.max(old_rate)
                if max_old_rate*dtmax < eps:
                    dt = dtmax
                else:
                    dt = max(eps/max_old_rate,dtmin)
                
                Zt = single_increment(Zt, dt, m, theta, sigma)

                new_rate = self.div_rate(Zt[:, 0], np.exp(Zt[:, 1]))
                Ht = Ht + dt/2 * (new_rate + old_rate)
                t  = t + dt
                time_since_last_obs += dt
                old_rate = new_rate

                indices_cell_divided = np.where(Ht > H)[0]
                number_of_divisions = len(indices_cell_divided)

                if time_since_last_obs > dtobs:
                    snapshots.append(Zt.copy())
                    times.append(t)
                    time_since_last_obs = 0
                else:
                    pass
            list_offs1, list_offs2 = [], []
        
            for mother_index in indices_cell_divided:
                mother_rate, mother_size = Zt[mother_index,0], np.exp(Zt[mother_index,1])

                offspring1_size, offspring1_rate = self.div_kernel(mother_size),mother_rate
                offspring2_size, offspring2_rate = mother_size - offspring1_size,mother_rate
                list_offs1.append((
                mother_rate,
                np.log(offspring1_size)))

                list_offs2.append((
                mother_rate,
                np.log(offspring2_size)))

            # Reuse past values injecting first offspring at the indexes of the cell that divided
            Zt[indices_cell_divided,:] = list_offs1
            Ht[indices_cell_divided] = 0.0
            
            H[indices_cell_divided] = np.random.exponential(1,size = number_of_divisions)

            # Adding the second offsprings
            Zt = np.concatenate((Zt, np.asarray(list_offs2)), axis = 0)
            Ht = np.concatenate((Ht, np.zeros(number_of_divisions)))
            H = np.concatenate((H, np.random.exponential(1,size = number_of_divisions)))
        
            Nt = len(Zt)
        return times, snapshots
            
    



if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    growth_rate = 1
    theta = np.sqrt(2)
    sigma = 1
    init_birth_size = 1

    def division_rate(r, x):
        return np.maximum(r, 0) * x

    def division_kernel(x):
        return x / 2

    ou_growth_rate_parameters = (
        growth_rate,
        theta,
        sigma
    )

    # State convention: Z = (R, log X)
    init = (
        growth_rate,
        init_birth_size
    )

    # Numerical parameters
    epsilons = [1]

    n_runs = 100

    dtmin = 1e-6
    dtmax = 1e-2
    dtobs = 1e-2

    max_cells = int(2*1e4)
    Tmax = 10

    # Fraction of the trajectory used to estimate the asymptotic slope
    tail_fraction = 0.25

    slopes = {
        eps: np.zeros(n_runs)
        for eps in epsilons
    }

    for eps in epsilons:

        print(f"\nTreating epsilon = {eps}")

        for run in range(n_runs):

            Model = Branching_OUGF(
                ou_growth_rate_parameters=ou_growth_rate_parameters,
                division_rate=division_rate,
                division_kernel=division_kernel
            )

            times, snapshots = Model.run(
                Tmax,
                init,
                dtmin=dtmin,
                dtmax=dtmax,
                dtobs=dtobs,
                eps=eps,
                max_number_of_cells=max_cells
            )

            times = np.asarray(times)

            # Total population mass
            mass = np.array([
                np.sum(np.exp(Zt[:, 1]))
                for Zt in snapshots
            ])

            log_mass = np.log(mass)

            # Use only the final fraction of the trajectory
            i0 = int(
                (1 - tail_fraction) * len(times)
            )

            tail_times = times[i0:]
            tail_log_mass = log_mass[i0:]

            # Linear regression:
            # log(M_t) ~ lambda * t + constant
            slope, intercept = np.polyfit(
                tail_times,
                tail_log_mass,
                1
            )

            slopes[eps][run] = slope

            if (run + 1) % 10 == 0:
                print(
                    f"  run {run + 1:3d}/{n_runs} "
                    f"| current mean slope = "
                    f"{np.mean(slopes[eps][:run+1]):.5f}"
                )

    # --------------------------------------------------
    # Print summary
    # --------------------------------------------------

    print("\nSlope estimates\n")

    for eps in epsilons:
        values = slopes[eps]

        print(
            f"epsilon = {eps:7.4f} : "
            f"mean = {np.mean(values):.6f}, "
            f"std = {np.std(values, ddof=1):.6f}, "
            f"SEM = {np.std(values, ddof=1) / np.sqrt(n_runs):.6f}"
        )

    # --------------------------------------------------
    # Plot distributions of estimated slopes
    # --------------------------------------------------

    fig, ax = plt.subplots(dpi=200)

    data = [
        slopes[eps]
        for eps in epsilons
    ]

    ax.boxplot(
        data,
        tick_labels=[
            f"{eps:g}"
            for eps in epsilons
        ],
        showmeans=True
    )

    ax.set_xlabel(r"$\varepsilon$")
    ax.set_ylabel(r"Estimated asymptotic slope $\widehat{\lambda}$")
    ax.set_title(
        f"Late-time growth-rate estimates "
        f"({n_runs} runs per epsilon)"
    )

    plt.tight_layout()
    plt.show()


    # --------------------------------------------------
    # Mean slope +/- Monte Carlo standard error
    # --------------------------------------------------

    means = np.array([
        np.mean(slopes[eps])
        for eps in epsilons
    ])

    stds = np.array([
        np.std(slopes[eps], ddof=1)
        for eps in epsilons
    ])

    sems = stds / np.sqrt(n_runs)

    fig, ax = plt.subplots(dpi=200)

    ax.errorbar(
        epsilons,
        means,
        yerr=2 * sems,
        fmt="o-",
        capsize=4
    )

    ax.set_xscale("log")
    ax.invert_xaxis()

    ax.set_xlabel(r"$\varepsilon$")
    ax.set_ylabel(r"Mean estimated slope")
    ax.set_title(r"Mean slope $\pm 2$ Monte Carlo standard errors")

    plt.tight_layout()
    plt.show()
    


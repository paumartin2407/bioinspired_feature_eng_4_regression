import random
from copy import deepcopy
import numpy as np
import pandas as pd
from tqdm import trange
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from evaluation import train_model
from features import candidate_frame


def remaining_evaluations(optimizer):
    counter = getattr(optimizer, "evaluation_counter", None)
    return counter.remaining if counter is not None else float("inf")

class GeneticAlgorithm:
    """
    Operadores: Tournament, Uniform Crossover, BLX-Alpha, Gaussian Mutation, Elitism.
    """

    def __init__(
        self,
        Xt_df,
        Xv_df,
        yt,
        yv,
        r_synth=3,          # Número de features sintéticas a crear
        pop_size=60,
        n_gens=70,
        tourn_size=3,          # Tamaño del torneo
        cross_prob=0.8,        # Probabilidad de cruce
        mut_prob=0.3,          # Probabilidad de mutación
        mut_prob_min=0.05,     # Probabilidad min mutación
        blx_alpha=0.5,      # Factor de expansión para BLX-Alpha
        elitism=2,          # Individuos élite
        seed=42,
        model=None,
        verbose=True,
        algorithm="GA_BLX", mode="combined"
    ):
        self.Xt_df = Xt_df.copy()
        self.Xv_df = Xv_df.copy()
        self.yt = self.target_array(yt)
        self.yv = self.target_array(yv)

        self.r_synth = int(r_synth)
        self.pop_size = int(pop_size)
        self.n_gens = int(n_gens)
        self.tourn_size = int(tourn_size)
        self.cross_prob = float(cross_prob)
        self.mut_prob = float(mut_prob)
        self.mut_prob_min = float(mut_prob_min)
        self.blx_alpha = float(blx_alpha)
        self.elitism = int(elitism)
        self.verbose = bool(verbose)
        
        self.n_features = self.Xt_df.shape[1]
        self.feature_names = list(self.Xt_df.columns)

        self.seed = seed
        self.rng = np.random.RandomState(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)

        # Configuración del modelo evaluador
        if model is not None:
            self.model = model
        else:
            self.model = lambda: LinearRegression()
            
        self.algorithm = algorithm
        self.mode = mode

    @staticmethod
    def target_array(y):
        if isinstance(y, (pd.DataFrame, pd.Series)):
            return y.values
        return np.asarray(y)

    def make_individual(self, p_select=0.3):
        """Inicialización aleatoria."""
        # 1. Selección de columnas originales (Binario)
        mask = self.rng.binomial(1, p_select, size=self.n_features).astype(int)
        # 2. Asegurar al menos 1 feature seleccionada
        if np.sum(mask) == 0: mask[self.rng.randint(0, self.n_features)] = 1
            
        # 3. Pesos para crear features sintéticas
        coefs = self.rng.normal(0.0, 0.5, size=(self.r_synth, self.n_features))
        
        # 4. Lineal vs Cuadrático (Binario/Discreto)
        type_mask = self.rng.randint(0, 2, size=self.r_synth).astype(int)
        
        return {"mask": mask, "coefs": coefs, "type_mask": type_mask}

    def individual_to_dataframe(self, ind, X_df):
        if self.mode != "combined":
            return candidate_frame(ind, X_df, "GA", mode=self.mode)
        
        mask = np.asarray(ind["mask"]).astype(bool)
        
        # 1. Selección de originales
        selected = X_df.loc[:, mask].copy()
        if selected.shape[1] == 0: 
             selected = X_df.iloc[:, [0]]
        
        Xp = selected.copy()
        X_vals = X_df.values

        # 2. Creación de Sintéticas (Combinación lineal + No linealidad)
        for i in range(self.r_synth):
            z = X_vals @ ind["coefs"][i]
            # Aplicar no-linealidad si el gen lo dice
            if ind["type_mask"][i] == 1:
                z = np.power(z, 2)
            Xp[f"synth_{i}"] = z
        return Xp

    def fitness(self, ind):
        try:
            Xpt = self.individual_to_dataframe(ind, self.Xt_df)
            Xpv = self.individual_to_dataframe(ind, self.Xv_df)
            
            # Protección contra NaNs o Infinitos generados por potencias grandes
            if not np.isfinite(Xpt.values).all() or not np.isfinite(Xpv.values).all():
                return 1e9

            model = self.model()
            model.fit(Xpt.values, self.yt)
            y_pred = model.predict(Xpv.values)
            
            return float(mean_absolute_error(self.yv, y_pred))
        except:
            return 1e9

    # --- OPERADOR 1: TOURNAMENT SELECTION ---
    def tournament_selection(self, population, fitnesses):
        """Selecciona el mejor de k individuos al azar."""
        idxs = self.rng.choice(len(population), size=self.tourn_size, replace=False)
        candidates_fits = [fitnesses[i] for i in idxs]
        best_idx = idxs[np.argmin(candidates_fits)]
        return deepcopy(population[best_idx])

    # --- OPERADOR 2: BLX-ALPHA (Para Coeficientes Reales) ---
    def _blx_alpha_crossover(self, c1, c2):
        #1. Calcular intervalo d
        d = np.abs(c1 - c2)
        
        #2. Límites expandidos
        min_v = np.minimum(c1, c2) - self.blx_alpha * d
        max_v = np.maximum(c1, c2) + self.blx_alpha * d
        
        #3. Generar hijo aleatorio en el rango expandido
        child_vals = self.rng.uniform(min_v, max_v)
        return child_vals

    # --- OPERADOR 3: UNIFORM CROSSOVER (Para Máscaras Binarias) ---
    def crossover(self, p1, p2, mask_prob=0.5, coefs_alpha=0.5):
        child = deepcopy(p1)
        
        # 1. Uniform Crossover para la Máscara de Selección (Discreto)
        u_mask = self.rng.rand(len(p1["mask"])) < mask_prob
        child["mask"] = np.where(u_mask, p2["mask"], p1["mask"])
        
        # 2. Uniform Crossover para el Tipo de Función (Lineal/Cuadrático)
        u_type = self.rng.rand(len(p1["type_mask"])) < coefs_alpha
        child["type_mask"] = np.where(u_type, p2["type_mask"], p1["type_mask"])
        
        # 3. BLX-Alpha para los Coeficientes (Continuo)
        child["coefs"] = self._blx_alpha_crossover(p1["coefs"], p2["coefs"])
        
        return child

    # --- OPERADOR 4: MUTACIÓN (Híbrida) ---
    def mutate(self, ind):
        n_mask = len(ind["mask"])
        p_mut_mask = max(self.mut_prob_min, self.mut_prob / n_mask)
        # 1. Bit-Flip Mutation para Máscaras (Cambiar estado de feature)
        mask_mut = self.rng.rand(n_mask) < p_mut_mask
        ind["mask"][mask_mut] = 1 - ind["mask"][mask_mut]

        if np.sum(ind["mask"]) == 0:
            fix_idx = self.rng.randint(0, n_mask)
            ind["mask"][fix_idx] = 1
        
        # 2. Gaussian Mutation para Coeficientes (Ajuste fino)
        # Sumamos ruido normal solo a algunos coeficientes o a todos con intensidad baja
        noise = self.rng.normal(0.0, 0.2, size=ind["coefs"].shape)
        # Aplicamos mutación con prob mut_prob
        do_mutate = self.rng.rand(*ind["coefs"].shape) < self.mut_prob
        ind["coefs"][do_mutate] += noise[do_mutate]
        
        # 3. Bit-Flip para Tipo
        n_types = len(ind["type_mask"])
        p_mut_type = max(self.mut_prob_min, self.mut_prob / n_types)

        type_mut = self.rng.rand(n_types) < p_mut_type
        ind["type_mask"][type_mut] = 1 - ind["type_mask"][type_mut]
        
        return ind

    def run(self):
        # 1. Inicialización
        pop = [self.make_individual() for _ in range(self.pop_size)]
        fitnesses = [self.fitness(ind) for ind in pop]
        history = []
        
        best_global_fit = float('inf')
        best_global_ind = None

        if self.verbose:
            print(f"Iniciando GA ({self.n_gens} gens) con BLX-Alpha y Uniform Crossover...")

        for gen in range(self.n_gens):
            if remaining_evaluations(self) == 0:
                break
            new_pop = []
            
            # 2. Elitismo
            # Ordenamos por fitness (menor es mejor)
            idx_sorted = np.argsort(fitnesses)
            
            # 3. Guardamos al mejor global si supera lo visto
            current_best_fit = fitnesses[idx_sorted[0]]
            if current_best_fit < best_global_fit:
                best_global_fit = current_best_fit
                best_global_ind = deepcopy(pop[idx_sorted[0]])
            
            # 4. Pasamos los k mejores directamente a la siguiente generación
            for e in range(self.elitism):
                new_pop.append(deepcopy(pop[idx_sorted[e]]))

            # 5. Rellenamos el resto
            while len(new_pop) < self.pop_size:
                # Selección
                p1 = self.tournament_selection(pop, fitnesses)
                p2 = self.tournament_selection(pop, fitnesses)
                
                # Cruce
                if self.rng.rand() < self.cross_prob:
                    child = self.crossover(p1, p2)
                else:
                    child = deepcopy(p1) # Si no hay cruce, clonamos
                
                # Mutación
                if self.rng.rand() < self.mut_prob:
                    child = self.mutate(child)
                new_pop.append(child)

            evaluated = min(len(new_pop), remaining_evaluations(self))
            new_fitnesses = [self.fitness(ind) for ind in new_pop[:evaluated]]
            candidate_idx = int(np.argmin(new_fitnesses))
            if new_fitnesses[candidate_idx] < best_global_fit:
                best_global_fit = new_fitnesses[candidate_idx]
                best_global_ind = deepcopy(new_pop[candidate_idx])
            
            # Guardar histórico
            history.append(best_global_fit)
            
            if self.verbose and gen % 10 == 0:
                print(f"Gen {gen}: Best MAE = {best_global_fit:.4f}")

            if evaluated < len(new_pop):
                break
            pop, fitnesses = new_pop, new_fitnesses

        final_idx = int(np.argmin(fitnesses))
        if fitnesses[final_idx] < best_global_fit:
            best_global_fit = fitnesses[final_idx]
            best_global_ind = deepcopy(pop[final_idx])
        Xpt_best = self.individual_to_dataframe(best_global_ind, self.Xt_df)
        Xpv_best = self.individual_to_dataframe(best_global_ind, self.Xv_df)
        result = train_model(Xpt_best.values, self.yt, Xpv_best.values, self.yv, model=self.model())

        result.update({
            "algorithm": self.algorithm,
            "features_generated": Xpt_best.columns.tolist(),
            "best_individual": best_global_ind,
            "best_expr": str(best_global_ind),
            "best_fitness": float(best_global_fit),
            "X_train_optimized": Xpt_best,
            "X_test_optimized": Xpv_best,
            "history": history,
            "feature_names":self.feature_names
        })

        return result

class PSO:
    def __init__(self, X_train, X_test, y_train, y_test,
                 n_particles=45, n_iters=45, w=0.65, c1=1.5, c2=1.5, r_synth=3, seed=42, verbose=True, mode="combined"):
        self.Xt_df = X_train.copy()
        self.Xv_df = X_test.copy()
        self.yt = np.array(y_train)
        self.yv = np.array(y_test)
        
        self.n_particles = n_particles
        self.n_iters = n_iters

        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.r_synth = r_synth
        self.verbose = verbose
        self.mode = mode
        self.rng = np.random.default_rng(seed)
        
        self.feature_names = list(X_train.columns)
        self.n_features = X_train.shape[1]
        
        # Dimensión total: [Mascara] + [Coeficientes] + [Tipos]
        self.dim = self.n_features + r_synth * self.n_features + r_synth


    def particle_df(self, particle, X_df):
        n_features = self.n_features
        
        # 1. Decodificar Máscara
        mask = (1 / (1 + np.exp(-particle[:n_features]))) > 0.5
        if not np.any(mask): mask[0] = True
            
        selected = X_df.copy() if self.mode == "construction" else X_df.loc[:, mask]
        if self.mode == "selection":
            return selected.copy()
        
        # 2. Decodificar Coeficientes
        start = n_features
        end = n_features + self.r_synth * n_features
        coefs = particle[start:end].reshape(self.r_synth, n_features)
        
        # 3. Decodificar Tipos
        synth_type = (1 / (1 + np.exp(-particle[-self.r_synth:]))) > 0.5

        # 4. Construir DataFrame
        Xp = selected.copy()
        X_vals = X_df.values
        
        for i in range(self.r_synth):
            z = X_vals @ coefs[i]
            if synth_type[i]:
                z = z**2 
            Xp[f"synth_{i}"] = z
            
        return Xp

    def _evaluate_model_internal(self, X_train, X_test):
        try:
            if not np.isfinite(X_train.values).all() or not np.isfinite(X_test.values).all():
                return 1e6
            model = LinearRegression()
            model.fit(X_train, self.yt)
            y_pred = model.predict(X_test)
            return mean_absolute_error(self.yv, y_pred)
        except:
            return 1e6

    def fitness_particle(self, particle):
        return self._evaluate_model_internal(
            self.particle_df(particle, self.Xt_df),
            self.particle_df(particle, self.Xv_df)
        )

    def inspect_particle(self, particle):
        print(f"\n{'='*20} INSPECCIÓN DE LA MEJOR PARTÍCULA {'='*20}")
        
        n_features = self.n_features
        feature_names = self.feature_names
        
        # 1. Ver qué columnas originales seleccionó
        mask = (1 / (1 + np.exp(-particle[:n_features]))) > 0.5
        selected_features = [f for f, m in zip(feature_names, mask) if m]
        print(f"Features Originales Seleccionadas ({len(selected_features)}/{n_features}):")
        print(f"   {selected_features}")
        
        # 2. Ver las fórmulas sintéticas
        coefs = particle[n_features:n_features+self.r_synth*n_features].reshape(self.r_synth, n_features)
        synth_type = (1 / (1 + np.exp(-particle[-self.r_synth:]))) > 0.5
        
        print("\n Features Sintéticas Generadas:")
        for i in range(self.r_synth):
            coefs_sel = coefs[i][mask]
            
            top_idx = np.argsort(np.abs(coefs_sel))[-3:][::-1]
            
            formula_parts = []
            for j in top_idx:
                fname = selected_features[j]
                weight = coefs_sel[j]
                sign = "+" if weight >= 0 else "-"
                formula_parts.append(f"{sign} {abs(weight):.2f}*{fname}")
            
            formula_str = " ".join(formula_parts)
            type_str = "CUADRÁTICA (x²)" if synth_type[i] else "LINEAL"
            
            print(f" Synth_{i} [{type_str}]: ( {formula_str} ... )")
        print("="*60)

    def run(self):
        # Inicialización
        pos = self.rng.uniform(-1, 1, (self.n_particles, self.dim))
        vel = np.zeros_like(pos)
        
        pbest_pos = pos.copy()
        pbest_val = np.array([self.fitness_particle(p) for p in pos])
        
        gbest_idx = np.argmin(pbest_val)
        gbest_pos = pbest_pos[gbest_idx].copy()
        gbest_val = pbest_val[gbest_idx]

        history = []

        iterator = trange(self.n_iters, desc="PSO Running", disable=not self.verbose)
        for it in iterator:
            remaining = remaining_evaluations(self)
            if remaining == 0:
                break
            # Update Vectorizado
            r1, r2 = self.rng.random((self.n_particles, self.dim)), self.rng.random((self.n_particles, self.dim))
            vel = self.w * vel + self.c1 * r1 * (pbest_pos - pos) + self.c2 * r2 * (gbest_pos - pos)
            vel = np.clip(vel, -5, 5)
            pos += vel
            
            # Evaluación
            for i in range(min(self.n_particles, remaining)):
                fit = self.fitness_particle(pos[i])
                if fit < pbest_val[i]:
                    pbest_val[i] = fit
                    pbest_pos[i] = pos[i].copy()
                    if fit < gbest_val:
                        gbest_val = fit
                        gbest_pos = pos[i].copy()
            
            history.append(gbest_val)
            if self.verbose and it % 10 == 0:
                iterator.set_description(f"PSO Iter {it}: Best MAE {gbest_val:.4f}")

        if self.verbose:
            print(f"\n PSO Terminado. Mejor MAE: {gbest_val:.5f}")
            self.inspect_particle(gbest_pos)

        n_feat = self.n_features
        
        mask = (1 / (1 + np.exp(-gbest_pos[:n_feat]))) > 0.5
        
        start_c = n_feat
        end_c = n_feat + self.r_synth * n_feat
        coefs = gbest_pos[start_c:end_c].reshape(self.r_synth, n_feat)

        types = (1 / (1 + np.exp(-gbest_pos[-self.r_synth:]))) > 0.5
        
        best_ind_dict = {
            "mask": mask.astype(int),
            "coefs": coefs,
            "type_mask": types.astype(int)
        }

        result = train_model(X_train=self.particle_df(gbest_pos, self.Xt_df),
                                    X_test=self.particle_df(gbest_pos, self.Xv_df),
                                    y_train=self.yt,
                                    y_test=self.yv,
                                )


        result.update({
            "algorithm": "PSO",
            "features_generated": self.particle_df(gbest_pos, self.Xt_df).columns.tolist(),
            "best_expr": gbest_pos.tolist(),   # equivalente al "genome" o "expresión"
            "best_fitness": float(gbest_val),
            "best_individual": best_ind_dict,  
            "best_ind": best_ind_dict,
            "X_train_optimized": self.particle_df(gbest_pos, self.Xt_df),
            "X_test_optimized": self.particle_df(gbest_pos, self.Xv_df),
            "history": history,
            "feature_names": self.feature_names
        })

        return result

class EvolutionStrategy:
    def __init__(self, X_train, X_test, y_train, y_test,
                 mu=20, lambda_=60, n_gens=50,
                 mutation_max=0.2, r_synth=3, seed=42, verbose=True, mode="combined"):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        self.mu = mu
        self.lambda_ = lambda_
        self.n_gens = n_gens
        self.mutation_max = mutation_max
        self.r_synth = r_synth
        self.verbose = verbose
        self.mode = mode
        np.random.seed(seed)
        random.seed(seed)
        self.history = []

    # 1. Crear individuo
    def create_individual(self, features):
        mask = np.random.rand(features) > 0.5
        coef = np.random.randn(self.r_synth, features)
        type_mask = np.random.rand(self.r_synth) > 0.5
        return {'mask': mask, 'coefs': coef, 'type_mask': type_mask}

    # 2. Transformación de datos
    def individual_to_df(self, ind, X_df):
        return candidate_frame(ind, X_df, "ES", reference=self.X_train, mode=self.mode)

    # 3. Evaluación
    def fitness(self, ind):
        X_train_i = self.individual_to_df(ind, self.X_train)
        X_test_i = self.individual_to_df(ind, self.X_test)
        results = train_model(X_train_i, self.y_train, X_test_i, self.y_test)
        return results["mae_test"]

    # 4. Mutación adaptativa
    def mutate(self, ind, mutation_rate):
        n_features = len(ind['mask'])
        n_synth = ind['coefs'].shape[0]

        for i in range(n_features):
            if np.random.rand() < mutation_rate:
                ind['mask'][i] = not ind['mask'][i]

        ind['coefs'] += mutation_rate * np.random.randn(n_synth, n_features)

        for i in range(n_synth):
            if np.random.rand() < mutation_rate:
                ind['type_mask'][i] = not ind['type_mask'][i]

        return ind

    # 5. Recombinación
    def recombine(self, p1, p2):
        child = {}
        child['mask'] = np.logical_or(p1['mask'], p2['mask'])
        child['coefs'] = np.where(np.random.rand(*p1['coefs'].shape) < 0.5,
                                  p1['coefs'], p2['coefs'])
        child['type_mask'] = np.where(np.random.rand(len(p1['type_mask'])) < 0.5,
                                      p1['type_mask'], p2['type_mask'])
        return child

    def run(self):
        n_features = self.X_train.shape[1]
        population = [self.create_individual(n_features) for _ in range(self.mu)]
        best_ind = None
        best_mae = float('inf')
        iterator = trange(self.n_gens, desc="Evolving", disable=not self.verbose)

        for g in range(self.n_gens):
            remaining = remaining_evaluations(self)
            if remaining == 0:
                break
            mutation_rate = self.mutation_max * (1 - g / self.n_gens)

            offspring = []
            for _ in range(self.lambda_):
                p1, p2 = random.sample(population, 2)
                child = self.recombine(p1, p2)
                child = self.mutate(child, mutation_rate)
                offspring.append(child)

            all_individuals = population + offspring
            evaluated = min(len(all_individuals), remaining)
            fitnesses = [self.fitness(ind) for ind in all_individuals[:evaluated]]

            sorted_idx = np.argsort(fitnesses)
            population = [all_individuals[i] for i in sorted_idx[:self.mu]]

            if fitnesses[sorted_idx[0]] < best_mae:
                best_mae = fitnesses[sorted_idx[0]]
                best_ind = all_individuals[sorted_idx[0]]

            self.history.append(best_mae)
            if self.verbose and g % 10 == 0:
                iterator.set_description(f"Best MAE: {best_mae:.4f} ")
            if evaluated < len(all_individuals):
                break

    
        X_train_opt = self.individual_to_df(best_ind, self.X_train)
        X_test_opt = self.individual_to_df(best_ind, self.X_test)

        results = train_model(X_train_opt, self.y_train, X_test_opt, self.y_test,
                                )
        results.update({
            "best_individual": best_ind,
            "X_train_optimized": X_train_opt,
            "X_test_optimized": X_test_opt,
            "history": self.history,
            "feature_names": list(self.X_train.columns),
            "best_fitness": best_mae,
            "mask": best_ind["mask"].astype(int).tolist(),
            "n_synth": best_ind["coefs"].shape[0]
        })

        if self.verbose:
            print("\n Mejor individuo:")
            print(f"MAE: {results['mae']:.4f}")
        return results



class Node:
    def evaluate(self, X): pass
    def __str__(self): pass

class FeatureNode(Node):
    def __init__(self, idx, name):
        self.idx = idx
        self.name = name
    def evaluate(self, X):
        return X[:, self.idx] 
    def __str__(self): return self.name

class ConstantNode(Node):
    def __init__(self, val): self.val = val
    def evaluate(self, X): return np.full(X.shape[0], self.val)
    def __str__(self): return f"{self.val:.2f}"

class BinaryOpNode(Node):
    def __init__(self, left, right, symbol, func):
        self.left = left; self.right = right; self.symbol = symbol; self.func = func
    def evaluate(self, X):
        return self.func(self.left.evaluate(X), self.right.evaluate(X))
    def __str__(self): return f"({self.left} {self.symbol} {self.right})"

class UnaryOpNode(Node):
    def __init__(self, child, symbol, func):
        self.child = child; self.symbol = symbol; self.func = func
    def evaluate(self, X): return self.func(self.child.evaluate(X))
    def __str__(self): return f"{self.symbol}({self.child})"


def protected_div(a, b): return np.divide(a, b, out=np.ones_like(a), where=np.abs(b)>1e-3)
def protected_sqrt(a): return np.sqrt(np.abs(a))
def protected_log(a): return np.log(np.abs(a)+1e-3)

class GeneticProgrammingFeatureEngineer:
    def __init__(self, X_train, X_test, y_train, y_test,
                 pop_size=50, n_gens=30, max_depth=4, 
                 p_cross=0.7, p_mut=0.2, seed=42, verbose=False):
        
        self.Xt = X_train
        self.Xv = X_test
        self.yt = np.array(y_train).ravel()
        self.yv = np.array(y_test).ravel()
        
        self.pop_size = pop_size
        self.n_gens = n_gens
        self.max_depth = max_depth
        self.p_cross = p_cross
        self.p_mut = p_mut
        self.verbose = verbose
        
        self.rng = np.random.default_rng(seed)
        random.seed(seed)
        
        self.feat_names = list(X_train.columns)
        self.n_feat = X_train.shape[1]
        
        self.bin_ops = [('+', np.add), ('-', np.subtract), ('*', np.multiply), ('/', protected_div)]
        self.un_ops = [('sin', np.sin), ('cos', np.cos), ('sqrt', protected_sqrt)]

    # --- Generar Árbol ---
    def _grow(self, depth):
        if depth==0 or (depth<self.max_depth and self.rng.random()<0.3):
            if self.rng.random() < 0.8:
                idx = self.rng.integers(0, self.n_feat)
                return FeatureNode(idx, self.feat_names[idx])
            else: return ConstantNode(self.rng.uniform(-5, 5))
        
        if self.rng.random() < 0.7:
            s, f = random.choice(self.bin_ops)
            return BinaryOpNode(self._grow(depth-1), self._grow(depth-1), s, f)
        else:
            s, f = random.choice(self.un_ops)
            return UnaryOpNode(self._grow(depth-1), s, f)

    # --- Feature Construction + Evaluación ---
    def _evaluate_individual(self, tree):
        try:
            # 1. Generar la Feature Sintética (Vectorizado = Rápido)
            z_train = tree.evaluate(self.Xt.values)
            z_test = tree.evaluate(self.Xv.values)
            
            
            if not np.isfinite(z_train).all() or not np.isfinite(z_test).all(): return 1e9

            mean_train = np.mean(z_train)
            std_train = np.std(z_train)

            if std_train < 1e-9: 
                return 1e9 
            
            # 2. Normalización (Idea de tu código nuevo)
            z_train = (z_train - mean_train) / std_train
            z_test = (z_test - mean_train) / std_train

            # 3. Entrenar Modelo: Originales + Sintética
            X_tr_aug = np.column_stack((self.Xt.values, z_train))
            X_te_aug = np.column_stack((self.Xv.values, z_test))
            
            m = LinearRegression()
            m.fit(X_tr_aug, self.yt)
            y_pred = m.predict(X_te_aug)
            
            return mean_absolute_error(self.yv, y_pred)
            
        except: return 1e9

    # --- Operadores Genéticos (Estándar GP) ---
    @staticmethod
    def _nodes_with_depth(root):
        nodes = []
        def visit(node, depth):
            nodes.append((node, depth))
            if hasattr(node, "left"):
                visit(node.left, depth + 1)
                visit(node.right, depth + 1)
            elif hasattr(node, "child"):
                visit(node.child, depth + 1)
        visit(root, 0)
        return nodes

    @staticmethod
    def _tree_depth(node):
        if hasattr(node, "left"):
            return 1 + max(GeneticProgrammingFeatureEngineer._tree_depth(node.left),
                           GeneticProgrammingFeatureEngineer._tree_depth(node.right))
        if hasattr(node, "child"):
            return 1 + GeneticProgrammingFeatureEngineer._tree_depth(node.child)
        return 0

    def _crossover(self, p1, p2):
        child = deepcopy(p1)
        target, depth = random.choice(self._nodes_with_depth(child))
        donors = [node for node, _ in self._nodes_with_depth(p2)
                  if self._tree_depth(node) <= self.max_depth - depth]
        subtree = deepcopy(random.choice(donors))
        target.__class__ = subtree.__class__
        target.__dict__ = subtree.__dict__
        return child

    def _mutate(self, p):
        child = deepcopy(p)
        target, depth = random.choice(self._nodes_with_depth(child))
        replacement = self._grow(depth=min(2, self.max_depth - depth))
        target.__class__ = replacement.__class__
        target.__dict__ = replacement.__dict__
        return child

    # --- Ejecución ---
    def run(self):
        if self.verbose:
            print(f" GP Feature Engineer: {self.n_gens} gens, {self.pop_size} pop")
        pop = [self._grow(self.max_depth) for _ in range(self.pop_size)]
        fits = [self._evaluate_individual(t) for t in pop]
        
        best_glob_fit = float('inf')
        best_glob_tree = None
        history = []
        
        iterator = trange(self.n_gens, desc="Evolving", disable=not self.verbose)
        for g in iterator:
            if remaining_evaluations(self) == 0:
                break
            new_pop = []
            # Elitismo
            best_idx = np.argmin(fits)
            if fits[best_idx] < best_glob_fit:
                best_glob_fit = fits[best_idx]
                best_glob_tree = deepcopy(pop[best_idx])
            new_pop.append(best_glob_tree)
            history.append(best_glob_fit)
            
            # Torneo y Reproducción
            while len(new_pop) < self.pop_size:
                cand_idx = self.rng.integers(0, self.pop_size, 3)
                parent = pop[cand_idx[np.argmin([fits[i] for i in cand_idx])]]
                
                if self.rng.random() < self.p_cross:
                    mate_idx = self.rng.integers(0, self.pop_size, 3)
                    mate = pop[mate_idx[np.argmin([fits[i] for i in mate_idx])]]
                    child = self._crossover(parent, mate)
                elif self.rng.random() < self.p_mut: child = self._mutate(parent)
                else: child = deepcopy(parent)
                new_pop.append(child)
            
            evaluated = min(len(new_pop), remaining_evaluations(self))
            new_fits = [self._evaluate_individual(t) for t in new_pop[:evaluated]]
            if evaluated < len(new_pop):
                candidate_idx = int(np.argmin(new_fits))
                if new_fits[candidate_idx] < best_glob_fit:
                    best_glob_fit = new_fits[candidate_idx]
                    best_glob_tree = deepcopy(new_pop[candidate_idx])
                break
            pop, fits = new_pop, new_fits
            
            if g % 5 == 0: iterator.set_description(f"Best MAE: {best_glob_fit:.4f}")

        final_idx = int(np.argmin(fits))
        if fits[final_idx] < best_glob_fit:
            best_glob_fit = fits[final_idx]
            best_glob_tree = deepcopy(pop[final_idx])
        # Eval Final
        # Recreamos la data aumentada
        z_tr = best_glob_tree.evaluate(self.Xt.values)
        z_te = best_glob_tree.evaluate(self.Xv.values)
        mean, std = z_tr.mean(), z_tr.std()
        X_tr_aug = np.column_stack((self.Xt.values, (z_tr - mean) / std))
        X_te_aug = np.column_stack((self.Xv.values, (z_te - mean) / std))
        
        result = train_model(X_train=X_tr_aug,
                                    X_test=X_te_aug,
                                    y_train=self.yt,
                                    y_test=self.yv,
                                )
        
        result.update({
            "algorithm": "Genetic Programming (Feature Eng)",
            "best_individual": best_glob_tree,
            "best_expr": str(best_glob_tree),
            "history": history, 
            "best_ind": best_glob_tree,
            "feature_names": self.feat_names + ["GP_Synthetic_Feature"]
        })
        return result

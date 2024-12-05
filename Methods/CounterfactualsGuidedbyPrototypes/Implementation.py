import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


class CounterfactualGenerator:
    
    def __init__(self,fpred, ENC, AE=None):
        """Counterfactual instance generator using Algorithm 1 proposed byVan Looveren et al.[1]
            
          This class implements a method for finding counterfactual instances based on a black-box model, using prototypes and other parameters.
      
          references
              [1] A. Van Looveren and J. Klaise, “Interpretable Counterfactual Explanations Guided by Prototypes,” Lect. Notes Comput. Sci. (including Subser. Lect. Notes Artif. Intell. Lect. Notes Bioinformatics), vol. 12976 LNAI, pp. 650–665, 2021, doi: 10.1007/978-3-030-86520-7_40/FIGURES/5.

          Args:
              fpred (callable): The prediction function of the black box model(torch.nn.Module). It should accept a tensor and return the predictions in logits.
              ENC (torch.nn.Module): Encoder model that transforms instances into the latent space.
              AE (torch.nn.Module, optional): Full autoencoder model. If provided, it is used in the loss function to ensure plausibility of counterfactuals.    Defaults to None.
              beta (float, optional): Weight of the L1 and L2 norms in the loss function, promoting sparsity in the perturbations. Defaults to 0.1.
              theta (float, optional): Weight of the prototype loss term, guiding the counterfactual towards the nearest class prototype. Defaults to 0.5.
              c (float, optional): Weight of the prediction loss term, ensuring the counterfactual is classified as the target class. Defaults to 1.0.
              kappa (float, optional): Margin used in the prediction loss to control the confidence in classification. Defaults to 0.5.
              gamma (float, optional): Weight of the autoencoder reconstruction loss, promoting plausibility of counterfactuals. Defaults to 0.1.
              K (int, optional): Number of nearest neighbors used to define the prototype for each class. Defaults to 5.
              max_iter (int, optional): Maximum number of iterations for the optimizer. Defaults to 1000.
              lr (float, optional): Learning rate for the optimizer (e.g., Adam). Controls the step size during optimization. Defaults to 0.01.
        """
        self.fpred = fpred
        self.ENC = ENC
        self.AE = AE


    def loss_pred(self,fpred, x_cf,kappa, t0):
        """
        Computes the prediction loss L_pred for guiding the counterfactual instance towards a different class.

        The loss is defined as:
        L_pred = max(fpred(x_cf)[t0] - max_{i ≠ t0} fpred(x_cf)[i], -kappa)
    
        - fpred(x_cf)[t0]: The model's predicted probability (or logit) for the original class t0 on the counterfactual instance x_cf.
        - max_{i ≠ t0} fpred(x_cf)[i]: The highest predicted probability for any class other than t0.
        - -kappa: A lower bound to cap the loss.
    
        This loss encourages the model's prediction for the counterfactual to shift away from the original class t0 towards another class,
        by maximizing the difference between the original class probability and the highest alternative class probability.
        If the difference is less than -kappa, the loss is set to -kappa to control its influence during optimization.
        """
        outputs = fpred(x_cf)
        logits = outputs
        top2 = torch.topk(logits, 2).values.squeeze()
        if kappa > 0:
            kappa_tensor = torch.tensor(-kappa, device=top2.device, dtype=top2.dtype)
            L_pred = torch.max(top2[0] - top2[1], kappa_tensor)
        else:
            L_pred = outputs[0, t0]

        return L_pred

    def loss_l1_l2(self,delta):
        """
        Computes the Elastic Net regularization terms L1 and L2 for the perturbations.
    
        The terms are defined as:
        - L1 = ||delta||_1: Promotes sparsity by minimizing the number of non-zero elements in delta.
          This ensures that only a few features are altered to generate the counterfactual.
        - L2 = ||delta||_2: Penalizes large perturbations, encouraging smooth or small-magnitude changes
          in the counterfactual features.
        """
        L1 = torch.norm(delta, p=1)
        L2 = torch.norm(delta, p=2)
    
        return L1,L2
    
    def loss_AE(self,AE,x_cf,gamma):
        """
        Computes the autoencoder reconstruction loss L_AE.
    
        This loss penalizes counterfactual instances that are far from the data manifold
        by measuring the reconstruction error using an autoencoder.
    
        The loss is defined as:
        L_AE = gamma * || x_cf - AE(x_cf) ||_2
    
        - x_cf: The counterfactual instance (x_0 + delta).
        - AE(x_cf): The reconstruction of x_cf by the autoencoder.
        - gamma: Weighting factor controlling the influence of the reconstruction loss.
    
        Explanation:
        - If the counterfactual x_cf is close to the data distribution, the autoencoder
          will reconstruct it accurately, resulting in a small reconstruction error.
          Therefore, the loss L_AE will be small, and its influence on the optimization
          will be minimal.
        - If x_cf is far from the data distribution, the autoencoder will poorly reconstruct
          it, leading to a large reconstruction error. This increases L_AE, penalizing
          counterfactuals that are unrealistic or implausible.
    
        Note:
        - If no autoencoder is provided (AE is None), the loss is set to zero.
        """
        
        if AE is not None:
            x_recon = AE(x_cf)
            L_AE = gamma * torch.norm(x_cf - x_recon.squeeze(), p=2)
        else:
            L_AE = 0
    
        return L_AE
    
    def loss_proto(self,ENC,x_cf,proto_j,theta,p=2):
        """
        Computes the prototype loss L_proto to guide the counterfactual towards the target class prototype.
    
        The loss is defined as:
        L_proto = theta * || ENC(x_cf) - proto_j ||_p^2
    
        Args:
            ENC (callable): The encoder model that transforms instances into the latent space.
            x_cf (torch.Tensor): The counterfactual instance x_cf = x0 + delta.
            proto_j (torch.Tensor): The prototype of the target class j.
            theta (float): Weighting factor controlling the influence of the prototype loss.
            p (int, optional): Norm degree (default is 2 for Euclidean distance).
    
        Returns:
            L_proto (torch.Tensor): The computed prototype loss.
    
        Explanation:
            - ENC(x_cf): Encodes the counterfactual instance into the latent space.
            - proto_j: Represents the target class prototype in the latent space.
            - The loss term theta * || ENC(x_cf) - proto_j ||_p^2 encourages the counterfactual
              to be close to the prototype of the target class, ensuring that it lies within
              the distribution of the target class in the latent space.
        """
        
        x_cf_enc = ENC(x_cf)
        L_proto = theta * torch.norm(x_cf_enc - proto_j, p) ** 2
    
        return L_proto
    
    def loss_cgp(self,c, L_pred, beta, L1, L2,L_AE,L_proto,trust=None, trust_threshold=1.5):
        
        #imrpove possible
        # trust_penalty = 0
        # if trust is not None and trust < trust_threshold:
        #     trust_penalty = (trust_threshold - trust) ** 2
    
        loss = c * L_pred + beta * ( L1 +  L2) +  L_AE +  L_proto #+ trust_penalty
        
        return loss
        
    def verbose_print(self,verbose,iteration,max_iter,loss,L_pred,L1,L2,L_AE,L_proto,AE):

        if verbose:
          log_message = f"Iteration {iteration + 1}/{max_iter}: Total Loss = {loss.item():.6f}, " \
                        f"L_pred = {L_pred.item():.6f}, L1 = {L1.item():.6f}, L2 = {L2.item():.6f}, "
          if AE is not None:
              log_message += f"L_AE = {L_AE.item():.6f}, "
          log_message += f"L_proto = {L_proto.item():.6f}"
          print(log_message)


    def plot_loss(self, plot,losses):
      if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(losses, label="Loss")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title("Evolution of Loss during Optimization")
        plt.legend()
        plt.grid(True)
        plt.show()

    def found_explanation(self,x0, X, beta=0.1, theta=0.5, c=1.0, kappa=0.5, gamma=0.1, K=5, max_iter=1000, lr=0.01 , verbose=True, plot=True):

        losses = []
        device = x0.device
        X.to(device)
    
        #Step 3
        with torch.no_grad():
    
            X_preds = self.fpred(X).argmax(dim=1)
            x0_pred = self.fpred(x0).argmax(dim=1).item()
    
        classes = torch.unique(X_preds)
        Xi = {i.item(): X[X_preds == i] for i in classes}
    
    
    
        #Step 4
        proto = {}
        x0_enc = self.ENC(x0).detach()

        for i in Xi:
            Xi_i = Xi[i]

            with torch.no_grad():
                self.ENC.eval()
                encodings = self.ENC(Xi_i).detach()

            x0_enc_expanded = x0_enc.expand(encodings.size(0), -1, -1, -1)
            distances = torch.norm((encodings - x0_enc_expanded).flatten(1), dim=1)
            idx_sorted = distances.argsort()[:K]
            nearest_encodings = encodings[idx_sorted]
            proto[i] = nearest_encodings.mean(dim=0)
            if K < 10:
              K += 1
    
        #Step 5
        min_dist = float('inf')
        t0 = x0_pred
        for i in proto:
            if i != t0:
                dist = torch.norm(x0_enc - proto[i])
                if dist < min_dist:
                    min_dist = dist
                    j = i
        proto_j = proto[j]
    
        #Step 6
        delta = torch.zeros_like(x0, requires_grad=True)
        optimizer = optim.Adam([delta], lr=lr)
        failed_attempts = 0
    
        for iteration in range(max_iter):
            optimizer.zero_grad()

            x_cf = x0 + delta

            x_cf_clipped = torch.clamp(x_cf, 0, 1)

            L_pred = self.loss_pred(self.fpred, x_cf_clipped,kappa, t0)
    
            L1,L2 = self.loss_l1_l2(delta)

            L_AE = self.loss_AE(self.AE,x_cf,gamma)
    
            L_proto = self.loss_proto(self.ENC,x_cf_clipped,proto_j,theta)

            proto_t0 = proto[t0]
            #trust = trust_score(ENC, x_cf_clipped, proto_t0, proto_j)

            loss = self.loss_cgp(c, L_pred, beta, L1, L2,L_AE,L_proto)
    
            self.verbose_print(verbose,iteration,max_iter,loss,L_pred,L1,L2,L_AE,L_proto,self.AE)
    
            losses.append(loss.item())
    
            loss.backward()
            torch.nn.utils.clip_grad_norm_(delta, max_norm=1.0)
            optimizer.step()
    
            with torch.no_grad():
                pred_cf = self.fpred(x_cf_clipped).argmax(dim=1).item()
    
            if pred_cf == j:
                print(f"Contrafactual encontrado en la iteración {iteration}")
                break
    
            # else:
            #     failed_attempts += 1
            #     if failed_attempts % 50 == 0:
            #       c *= 1.2
    
        #7Step
        x_cf_final = x0 + delta.detach()
        x_cf_final = torch.clamp(x_cf_final, 0, 1)  # Asegurar que esté dentro de [0,1]
    
        self.plot_loss(plot, losses)
    
        return x_cf_final

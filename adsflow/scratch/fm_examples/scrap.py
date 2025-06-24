
"""

######################################################
# Model sampling
######################################################
print("Begin Inference")
x = torch.randn(3000, 2)
n_steps = 8
fig, axes = plt.subplots(1, n_steps + 1, figsize=(30, 4), sharex=True, sharey=True)
time_steps = torch.linspace(0, 1.0, n_steps + 1)


axes[0].scatter(x.detach()[:, 0], x.detach()[:, 1], s=10, alpha=0.50)
axes[0].set_title(f't = {time_steps[0]:.2f}')
axes[0].set_xlim(-2.0, 2.0)
axes[0].set_ylim(-2.0, 2.0)

for i in range(n_steps):
    x = flow.step(xt=x, t_start=time_steps[i], t_end=time_steps[i + 1]) # sample xt from neural network 
    axes[i + 1].scatter(x.detach()[:, 0], x.detach()[:, 1], s=10, alpha=0.50)
    axes[i + 1].set_title(f't = {time_steps[i + 1]:.2f}')

# save as .svg
plt.savefig('q_p.svg', format='svg', dpi=300, bbox_inches='tight')

######################################################
# Generate p0 (checker board)
######################################################
# Generate data for a dummy distribution (testing purposes only)
rv1 = torch.rand(10000) * 4 - 2 
rv2 = torch.rand(10000) * 4 - 2 
mask = (torch.floor(rv1) + torch.floor(rv2)) % 2 == 0  

x1 = torch.stack((rv1[mask], rv2[mask]), dim=1)
x1_np = x1.numpy()

# Plot the checkerboard distribution
plt.figure(figsize=(4, 4))
plt.scatter(x1_np[:, 0], x1_np[:, 1], s=10, alpha=0.50)
plt.title("Checkerboard Distribution")
plt.xlabel("X1")
plt.ylabel("X2")
plt.xlim(-2, 2)
plt.ylim(-2, 2)
# save as .svg
plt.savefig('p_grnd.svg', format='svg', dpi=300, bbox_inches='tight')

plt.show()

"""

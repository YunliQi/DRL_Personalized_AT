import torch
import torch.nn.functional as F
import random

# Create a 5x5 tensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
grid = torch.arange(1, 26).view(5, 5).float().to(device)  # Use float for later operations

padded_grid = F.pad(grid, (1, 1, 1, 1), 'constant', -1)

# Unfold the grid to extract 3x3 patches
# The first unfold operation will roll along rows, and the second one along columns
patches = padded_grid.unfold(0, 3, 1).unfold(1, 3, 1)

# Reshape the patches to have each patch as a separate element in the batch dimension
patches = patches.contiguous().view(-1, 3*3)

columns_to_keep = [1, 3, 4, 5, 7]

# Select the desired columns
patches = patches[:, columns_to_keep]

# def customised_function(patch):
def perform_checks(patches: torch.Tensor, self, dose):
    device = patches.device
    test_4_rand = torch.rand(patches.size(0), 1, device=device)
    test_5_rand = torch.rand(patches.size(0), 1, device=device)
    test_6_rand = torch.rand(patches.size(0), 1, device=device)
    test_7_rand = torch.rand(patches.size(0), 1, device=device)
    test_9_rand = torch.rand(patches.size(0), 1, device=device)

    test_4_thresh = torch.tensor(self.r_s * (1 + self.d_cell)).to(device)
    test_5_thresh = torch.tensor(self.r_s / (self.r_s + self.r_s * (1 + self.d_cell))).to(device)
    test_6_thresh = torch.tensor((1 - self.c_r) * self.r_s + self.r_s * self.d_cell).to(device)
    test_7_thresh = torch.tensor((1 - self.c_r) * self.r_s / ((1 - self.c_r) * self.r_s + (1 - self.c_r) * self.r_s + self.r_s * self.d_cell)).to(device)
    test_9_thresh = torch.tensor(dose * self.d_drug).to(device)

    bool_checks = torch.zeros((patches.size(0), 9), device=device)


    bool_checks[:, 0] = patches[:, 2] == 0
    bool_checks[:, 1] = patches[:, 2] == 1
    bool_checks[:, 2] = patches[:, 2] == 2
    bool_checks[:, 3] = (test_4_rand < test_4_thresh).squeeze()
    bool_checks[:, 4] = (test_5_rand < test_5_thresh).squeeze()
    bool_checks[:, 5] = (test_6_rand < test_6_thresh).squeeze()
    bool_checks[:, 6] = (test_7_rand < test_7_thresh).squeeze()
    bool_checks[:, 7] = (patches[:, [0, 1, 3, 4]] == 0).any(dim=1)
    bool_checks[:, 8] = (test_9_rand < test_9_thresh).squeeze()

    change = torch.zeros((self.side_len ** 2), device=device)
    change.fill_(-1)

    pattern_1D_mask = (bool_checks[:, 1] == True) & (bool_checks[:, 3] == True) & (bool_checks[:, 4] == False)
    pattern_1PK_mask = (bool_checks[:, 1] == True) & (bool_checks[:, 3] == True) & (bool_checks[:, 4] == True) & (bool_checks[:, 7] == True) & (bool_checks[:, 8] == True)
    pattern_2D_mask = (bool_checks[:, 2] == True) & (bool_checks[:, 5] == True) & (bool_checks[:, 6] == False)

    pattern_1PNK_mask = (bool_checks[:, 1] == True) & (bool_checks[:, 3] == True) & (bool_checks[:, 4] == True) & (bool_checks[:, 7] == True) & (bool_checks[:, 8] == False)
    pattern_2P_mask = (bool_checks[:, 2] == True) & (bool_checks[:, 5] == True) & (bool_checks[:, 6] == True) & (bool_checks[:, 7] == True)

    pattern_change_dead = pattern_1D_mask | pattern_1PK_mask | pattern_2D_mask
    pattern_change_s = pattern_1PNK_mask
    pattern_change_r = pattern_2P_mask

    change[pattern_change_dead] = 0
    change[pattern_change_s] = 1
    change[pattern_change_r] = 2

    indices_dead = torch.nonzero(change == 0).to(device)
    if indices_dead.numel():

        row_num_dead = indices_dead // self.side_len
        col_num_dead = indices_dead % self.side_len

        dead_cells = torch.stack((row_num_dead[0], col_num_dead[0]), dim=1)
    else:
        dead_cells = torch.tensor([]).to(device)

    indices_s = torch.nonzero(change == 1).squeeze().to(device)
    neighbour_pos = [0, 1, 3, 4]

    if indices_s.numel():
        pro_s_cells = patches[indices_s]
        if pro_s_cells.dim() == 1:
            pro_s_cells = pro_s_cells.unsqueeze(0)
        
        pro_s_neighbour = pro_s_cells[:, neighbour_pos]
        avail_mask = pro_s_neighbour == 0

        true_indices = avail_mask.nonzero(as_tuple=True)
        rows, cols = true_indices

        # Step 2: For each row, randomly select one of the True entries
        unique_rows = torch.unique(rows, sorted=True)
        rand_choice_indices = torch.stack([torch.multinomial((rows == r).float(), 1) for r in unique_rows]).squeeze()

        # Extract the corresponding column indices for the randomly selected True values
        selected_cols = cols[rand_choice_indices]

        selected_cols = torch.where(selected_cols == 0, torch.tensor(-self.side_len).to(selected_cols.device), selected_cols)
        selected_cols = torch.where(selected_cols == 1, torch.tensor(-1).to(selected_cols.device), selected_cols)
        selected_cols = torch.where(selected_cols == 2, torch.tensor(1).to(selected_cols.device), selected_cols)
        selected_cols = torch.where(selected_cols == 3, torch.tensor(self.side_len).to(selected_cols.device), selected_cols)

        indices_s = indices_s + selected_cols

        row_num_s = indices_s // self.side_len
        col_num_s = indices_s % self.side_len

        if indices_s.numel() == 1:
            row_num_s = row_num_s.unsqueeze(0)
            col_num_s = col_num_s.unsqueeze(0)

        s_cells = torch.stack((row_num_s, col_num_s), dim=1)
    else:
        s_cells = torch.tensor([]).to(device)

    indices_r = torch.nonzero(change == 2).squeeze().to(device)

    if indices_r.numel():
        pro_r_cells = patches[indices_r]
        if pro_r_cells.dim() == 1:
            pro_r_cells = pro_r_cells.unsqueeze(0)
        pro_r_neighbour = pro_r_cells[:, neighbour_pos]
        avail_mask = pro_r_neighbour == 0

        true_indices = avail_mask.nonzero(as_tuple=True)
        rows, cols = true_indices

        # Step 2: For each row, randomly select one of the True entries
        unique_rows = torch.unique(rows, sorted=True)
        rand_choice_indices = torch.stack([torch.multinomial((rows == r).float(), 1) for r in unique_rows]).squeeze()

        # Extract the corresponding column indices for the randomly selected True values
        selected_cols = cols[rand_choice_indices]

        selected_cols = torch.where(selected_cols == 0, torch.tensor(-self.side_len).to(selected_cols.device), selected_cols)
        selected_cols = torch.where(selected_cols == 1, torch.tensor(-1).to(selected_cols.device), selected_cols)
        selected_cols = torch.where(selected_cols == 2, torch.tensor(1).to(selected_cols.device), selected_cols)
        selected_cols = torch.where(selected_cols == 3, torch.tensor(self.side_len).to(selected_cols.device), selected_cols)

        indices_r = indices_r + selected_cols

        row_num_r = indices_r // self.side_len
        col_num_r = indices_r % self.side_len

        if indices_r.numel() == 1:
            row_num_r = row_num_r.unsqueeze(0)
            col_num_r = col_num_r.unsqueeze(0)

        r_cells = torch.stack((row_num_r, col_num_r), dim=1)
    else:
        r_cells = torch.tensor([]).to(device)

    dead_list = dead_cells.cpu().detach().tolist()
    dead_tuples = [tuple(element) for element in dead_list]

    s_list = s_cells.cpu().detach().tolist()
    s_tuples = [tuple(element) for element in s_list]

    r_list = r_cells.cpu().detach().tolist()
    r_tuples = [tuple(element) for element in r_list]

    s_set = set(s_tuples)
    r_set = set(r_tuples)

    s_diff = s_set - r_set
    r_diff = r_set - s_set

    intersection = list(s_set & r_set)

    res = {}

    for dead_tuple in dead_tuples:
        res[dead_tuple] = 0

    for s_tuple in s_diff:
        res[s_tuple] = 1

    for r_tuple in r_diff:
        res[r_tuple] = 2

    for int_tuple in intersection:
        res[int_tuple] = random.randint(1, 2)

    return res

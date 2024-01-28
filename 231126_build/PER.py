import numpy as np

class SumTree:
    
    # Here we initialize the tree with all nodes = 0, and initialize the data with all values = 0
    def __init__(self, capacity):
        # Number of leaf nodes (final nodes) that contains experiences
        self.capacity = capacity
        self.data_pointer = 0
        # Generate the tree with all nodes values = 0
        # To understand this calculation (2 * capacity - 1) look at the schema below
        # Remember we are in a binary node (each node has max 2 children) so 2x size of leaf (capacity) - 1 (root node)
        # Parent nodes = capacity - 1
        # Leaf nodes = capacity
        self.tree = np.zeros(2 * capacity - 1)

        # Contains the experiences (so the size of data is capacity)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0


    def add(self, priority, data):
        # Look at what index we want to put the experience
        tree_index = self.data_pointer + self.capacity - 1

        # Update data frame
        self.data[self.data_pointer] = data

        # Update the leaf
        self.update (tree_index, priority)

        # Add 1 to data_pointer
        self.data_pointer += 1

        if self.data_pointer >= self.capacity:  # If we're above the capacity, we go back to first index (we overwrite)
            self.data_pointer = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1
        
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left_child_index = 2 * idx + 1
        right_child_index = left_child_index + 1

        if left_child_index >= len(self.tree):
            return idx
        
        if s <= self.tree[left_child_index]:
            return self._retrieve(left_child_index, s)
        else:
            return self._retrieve(right_child_index, s - self.tree[left_child_index])

    def get_leaf(self, s):
        leaf_index = self._retrieve(0, s)

        data_index = leaf_index - self.capacity + 1

        return leaf_index, self.tree[leaf_index], self.data[data_index]

    def total_priority(self):
        return self.tree[0] # Returns the root node

    # Update the leaf priority score and propagate the change through tree
    def update(self, tree_index, priority):
        # Change = new priority score - former priority score
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority

        # then propagate the change through tree
        # this method is faster than the recursive loop in the reference code
        self._propagate(tree_index, change)


# Now we finished constructing our SumTree object, next we'll build a memory object.
class PrioritizedReplayBuffer(object):  # stored as ( s, a, r, s_ ) in SumTree
    
    def __init__(self, capacity):   
        # Making the tree 
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.PER_e = 1e-8 # 어떤 경험을 할 확률이 0이 되는 것을 피하기 위해 사용하는 
        self.PER_a = 0.6  # 우선순위 높은 경험 선택과 무작위 샘플링 사이에 절충점 위해
        self.PER_b = 0.3  # 중요도 샘플링, 초기값에서 1로 증가 0.4

        self.PER_b_increment_per_sampling = 0.000001 #0.000002
        self.priorities = []
        self.minibatch = []

    def reset(self):
        self.tree = SumTree(self.capacity)

    def _getPriority(self, error):
        # print("error: ", error)
        return (error + self.PER_e) ** self.PER_a
    
    def __getitem__(self, idx):
        return self.tree.data[idx]

    def store(self, error, sample):
        max_priority = self._getPriority(error)
        #print("store error: ", max_priority)
        self.tree.add(max_priority, sample)

    # Now we create sample function, which will be used to pick batch from our tree memory, which will be used to train our model.
    # - First, we sample a minibatch of n size, the range [0, priority_total] into priority ranges.
    # - Then a value is uniformly sampled from each range.
    # - Then we search in the sumtree, for the experience where priority episode_reward correspond to sample values are retrieved from.
    def sample(self, n):
        # Create a minibatch array that will contains the minibatch
        idxs = []
        self.minibatch.clear()
        self.priorities.clear()
        priority_segment = self.tree.total_priority() / n
        
        self.PER_b = np.min([1., self.PER_b + self.PER_b_increment_per_sampling])

        i = 0
        while len(self.minibatch) < n:
            # 각 범위에서 균등하게 값을 샘플링합니다.
            a = priority_segment * i
            b = priority_segment * (i + 1)
            value = np.random.uniform(a, b)

            # 해당 값에 대응하는 경험을 가져옵니다.
            idx, p, data = self.tree.get_leaf(value)

            # 우선순위가 0이 아닌 경우에만 추가합니다.
            if p != 0:
                self.priorities.append(p)
                self.minibatch.append(data)
                idxs.append(idx)

            i += 1

        # print("tree.n_entries: ",self.tree.n_entries) sample 개수만큼 증가하는
        # print("priorities: ", priorities) 내부 요소가 0이 되는 경우가 발생하면 오류!!!!!
        # print("tree.total_priority: ", self.tree.total_priority())
        sampling_probabilities = self.priorities / self.tree.total_priority()
        #print(" sampling_probabilities :", sampling_probabilities) 문제 없음 
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, 1.0/self.PER_b)
        is_weight = np.clip(is_weight, 0, 1.0) # 0 ~ 1 범위로 제한 

        # minibatch는 numpy배열 
        return self.minibatch, idxs, is_weight

    # Update the priorities on the tree
    def batch_update(self, idx, error):
        p = self._getPriority(error)
        self.tree.update(idx, p)

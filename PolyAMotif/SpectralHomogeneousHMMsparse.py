import numpy as np
import scipy.sparse as sp
import scipy
import scipy.sparse.linalg


def _sparse_svd(X, k):
  """U, s, V = svd(X), only returns U."""
  # get indices of non-zero elements.
  X = X.tocoo()
  X_nrows, X_ncols = X.shape
  print 'original X shape ', X.shape
  row_indices = X.row
  column_indices = X.col

  # create hash map.
  row_hash = dict()
  row_counter = 0
  for row_index in row_indices:
    if row_index not in row_hash:
      row_hash[row_index] = row_counter
      row_counter += 1
  column_hash = dict()
  column_counter = 0
  for column_index in column_indices:
    if column_index not in column_hash:
      column_hash[column_index] = column_counter
      column_counter += 1

  print 'row_counter ', row_counter
  print 'column_counter ', column_counter
  # create small block non-zero matrix
  reduced_row_indices = [row_hash[row] for row in row_indices]
  reduced_column_indices = [column_hash[column] for column in column_indices]
  reduced_matrix = sp.coo_matrix((X.data,
                                  (reduced_row_indices,
                                   reduced_column_indices))).tocsc()
  print 'reduced_matrix shape ', reduced_matrix.shape
  
  # qr and svd.
  reduced_U, _s, _ = sp.linalg.svds(reduced_matrix, k)
  # Only care about column space not the order of the columns.
  # Desceding sort.
  sort_index = np.argsort(_s)[::-1]
  reduced_U = reduced_U[:, sort_index]
  print 'reduced_U shape ', reduced_U.shape
  
  # fill back to get U.
  reverse_row_hash = dict()
  for original_row, permuted_row in row_hash.iteritems():
    reverse_row_hash[permuted_row] = original_row
  
  # Since flatten() is row-based.
  temp_row, temp_col = np.mgrid[0:row_counter, 0:k]
  reverse_row_indices = [reverse_row_hash[row]
                         for row in temp_row.flatten()]
  # Don't care about column permutation because it does not
  # affect U.
  reverse_column_indices = temp_col.flatten()

  print '# of row indices ', len(reverse_row_indices)
  print '# of column indices ', len(reverse_column_indices)
  print '# of data elements ', len(reduced_U.flatten())
  U = sp.coo_matrix((reduced_U.flatten(),
                     (reverse_row_indices,
                      reverse_column_indices)),
                     shape=(X_nrows, k)).tocsc()
  print 'Output of svd shape ', U.shape

  return U


def _sparse_svd_dense(X, k):
  """U, s, V = svd(X), only returns U."""
  # get indices of non-zero elements.
  U, _, _ = np.linalg.svd(X.todense())

  return sp.csc_matrix(U[:, :k])


def _sparse_svd_shuffle_dense(X, k):
  """U, s, V = svd(X), only returns U."""
  # get indices of non-zero elements.
  X = X.tocoo()
  X_nrows, X_ncols = X.shape
  print 'original X shape ', X.shape
  row_indices = X.row
  column_indices = X.col

  # create hash map.
  row_hash = dict()
  row_counter = 0
  for row_index in row_indices:
    if row_index not in row_hash:
      row_hash[row_index] = row_counter
      row_counter += 1
  column_hash = dict()
  column_counter = 0
  for column_index in column_indices:
    if column_index not in column_hash:
      column_hash[column_index] = column_counter
      column_counter += 1

  print 'row_counter ', row_counter
  print 'column_counter ', column_counter
  # create small block non-zero matrix
  reduced_row_indices = [row_hash[row] for row in row_indices]
  reduced_column_indices = [column_hash[column] for column in column_indices]
  reduced_matrix = sp.coo_matrix((X.data,
                                  (reduced_row_indices,
                                   reduced_column_indices))).tocsc()
  print 'reduced_matrix shape ', reduced_matrix.shape
  
  # qr and svd.
  reduced_U, _, _ = np.linalg.svd(reduced_matrix.todense(), full_matrices=False)
  reduced_U = np.array(reduced_U[:, :k])
  print 'reduced_U shape ', reduced_U.shape
  
  # fill back to get U.
  reverse_row_hash = dict()
  for original_row, permuted_row in row_hash.iteritems():
    reverse_row_hash[permuted_row] = original_row
  
  # Since flatten() is row-based.
  temp_row, temp_col = np.mgrid[0:row_counter, 0:k]
  reverse_row_indices = [reverse_row_hash[row]
                         for row in temp_row.flatten()]
  # Don't care about column permutation because it does not
  # affect U.
  reverse_column_indices = temp_col.flatten()

  print '# of row indices ', len(reverse_row_indices)
  print '# of column indices ', len(reverse_column_indices)
  print '# of data elements ', len(reduced_U.flatten())
  U = sp.coo_matrix((reduced_U.flatten(),
                     (reverse_row_indices,
                      reverse_column_indices)),
                     shape=(X_nrows, k)).tocsc()
  print 'Output of svd shape ', U.shape

  return U


def _sparse_pinv_original(A):
  m, n = A.shape
  if m > n:
    small_block = A.T * A
  else:
    small_block = A * A.T

  machine_prec = np.finfo(np.float).eps * 20
  # Make sure it is symmetric for numerical reasons.
  small_block = (small_block + small_block.T) / 2
  small_eigv, small_eigU = np.linalg.eigh(small_block.todense())

  if np.iscomplexobj(small_eigU):
    print small_eigv, 'is complex'
    raise KeyError

  eigv_keep_index = np.abs(small_eigv) > machine_prec
  filtered_eigv = small_eigv[eigv_keep_index]
  filtered_eigU = small_eigU[:, eigv_keep_index]

  small_block_inverse = np.dot(filtered_eigU / filtered_eigv, filtered_eigU.T)
  small_block_inverse_sparse = sp.csc_matrix(small_block_inverse)
  if m > n:
    pinv_A = small_block_inverse_sparse * A.T
  else:
    pinv_A = A.T * small_block_inverse_sparse

  return pinv_A


def _sparse_pinv(A):
  m, n = A.shape
  if m > n:
    small_block = A.T * A
  else:
    small_block = A * A.T

  small_block_inverse = np.linalg.pinv(small_block.todense())
  small_block_inverse_sparse = sp.csc_matrix(small_block_inverse)
  if m > n:
    pinv_A = small_block_inverse_sparse * A.T
  else:
    pinv_A = A.T * small_block_inverse_sparse

  return pinv_A


def _NormalizeSparseCountMatrix(matrix_to_normalize):
  csc_format = matrix_to_normalize.tocsc()
  normalizer = csc_format.sum()
  if normalizer > 0.0:
    matrix_to_normalize = csc_format / csc_format.sum()
  else:
    matrix_to_normalize = csc_format

  # Conver to coo to save memory.
  return matrix_to_normalize.tocoo()


def _AllocateStatistics(i, current_counter, num_states):
#  print i
  if i % 50 == 0:
    print '%i: current_counter %i' % (i, len(current_counter))
  P_3x1 = dict()
  P_21 = sp.dok_matrix((num_states, num_states))
  P_1 = sp.dok_matrix((num_states, 1))
  for key, value in current_counter.items():
    x1, x2, x3 = key

    if x2 in P_3x1:
      P_3x1[x2][x3, x1] += value
    else:
      P_3x1[x2] = sp.dok_matrix((num_states, num_states))

    P_21[x2, x1] += value
    P_1[x1, 0] += value

  # Convert to csc for pickling issues.
  picklable_P_3x1 = dict()
  total_P_3x1_nnz = 0
  for key, val in P_3x1.iteritems():
    picklable_P_3x1[key] = val.tocoo()
    total_P_3x1_nnz += val.nnz
  picklable_P_21 = P_21.tocoo()
  picklable_P_1 = P_1.tocoo()

  total_elements = num_states * num_states * len(P_3x1)
  print 'P_3x1 sparsity %i/%i=%f ' % (total_P_3x1_nnz,
                                      total_elements,
                                      float(total_P_3x1_nnz)/total_elements)
  total_P_21_nnz = P_21.nnz
  total_P_21_elements = num_states * num_states
  print 'P_21 sparsity %i/%i=%f ' % (total_P_21_nnz,
                                     total_P_21_elements,
                                     float(total_P_21_nnz)/total_P_21_elements)
  total_P_1_nnz = P_1.nnz
  total_P_1_elements = num_states
  print 'P_1 sparsity %i/%i=%f ' % (total_P_1_nnz,
                                    total_P_1_elements,
                                    float(total_P_1_nnz)/total_P_1_elements)



  return picklable_P_3x1, picklable_P_21, picklable_P_1


def CalculateStatistics(train_sequences, num_states):
  # Scans through all triples in the sequences
  # to form a fourth-order tensor.
  sequence_length = len(train_sequences[0])

  # One scan to accumulate statistics.
  print 'Scan and accumulate statistics.'
  triple_counts = dict()
  for each_sequence in train_sequences:
    for i in xrange(sequence_length-2):
      x1, x2, x3 = each_sequence[i:i+3]
      key = (x1, x2, x3)
      triple_counts[key] = triple_counts.get(key, 0.0) + 1.0

  # Construct P_3x1 from the third order tensor.
  print 'P_3x1, P_21, P_1'
  P_3x1, P_21, P_1 = _AllocateStatistics(0, triple_counts, num_states)

  for x in P_3x1.iterkeys():
    P_3x1[x] = _NormalizeSparseCountMatrix(P_3x1[x])
  P_21 = _NormalizeSparseCountMatrix(P_21)
  P_1 = _NormalizeSparseCountMatrix(P_1)

  return P_1, P_21, P_3x1


def SpectralHMMTrain(train_sequences, num_states, k):
  # Compute first, second, and third order statistics
  # from training sequences. The following quantities
  # relate to 3.1 Definition from the paper.
  print 'Calculating stastics.'
  P_1, P_21, P_3x1 = CalculateStatistics(train_sequences, num_states)

  print 'Calculating observation quantities.'

  # Perform an SVD on P_21 to get U.
  print 'svd'
  U = _sparse_svd(P_21, k)

  # Caculate b_1, b_inf and B_x
  print 'b_1 and b_inf'
  print 'P_1 shape ', P_1.shape
  b_1 = np.ones((k, ))
  print 'b_1 shape ', b_1.shape
  b_1.shape = k,

  b_inf = np.ones((k, ))
  b_inf.shape = k,

  print 'B_x'
  B_x = dict()
  tmp2 = U.T * P_21.tocsc()
  tmp2_pinv = _sparse_pinv(tmp2)
  for x in P_3x1.iterkeys():
    tmp1 = U.T * P_3x1[x].tocsc()
    tmp3 = tmp1 * tmp2_pinv
    B_x[x] = np.array(tmp3.todense()) + 1e-6
    if np.iscomplexobj(B_x[x]):
      print 'B_x[x] is complex!'
      raise KeyError

  return b_1, b_inf, B_x


def SpectralHMMBeliefs(sequence, model):
  b1, b_inf, B_x = model
  state_beliefs = b1
  k = b1.size
  if np.sum(state_beliefs) < 0:
    state_beliefs *= -1.0
  belief_list = [state_beliefs.copy()]
  for i, x in enumerate(sequence[:-2]):
    if x in B_x:
      tmp_belief = np.dot(B_x[x], state_beliefs)
    else:
      tmp_belief = np.ones(state_beliefs.shape) / k
    normalizer = np.dot(b_inf.T, tmp_belief)
    #print normalizer
    state_beliefs = tmp_belief / normalizer
    #print np.sum(state_beliefs)
    belief_list.append(state_beliefs.copy())

  return belief_list

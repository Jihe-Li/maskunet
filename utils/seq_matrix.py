def seq2matrix(seq, h, w):
    '''
    将sequence转换为matrix形式
    '''
    matrix = seq.reshape(seq.shape[0], h, w)
    return matrix

def matrix2seq(matrix):
    seq = matrix.rehape(matrix.shape[0], -1)
    return seq


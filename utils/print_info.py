def print_info(info):
    message = ('Epoch: {}/{}, Duration: {:.3f}s, \n  Train Loss: {:.4f},' 
               'Test Loss: {}, \n  Train Acc: {:.4f}, Test Acc: {}').format(
                   info['current_epoch'], info['epochs'], info['t_duration'],
                   info['train_loss'], info['test_loss'], info['train_acc'], info['test_acc'])
    print(message)


def print_message(info):
    message = ('Epoch: {}/{}, Duration: {:.3f}s, \n' 
               'Train Loss: {:.4f}, Test Loss: {:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f} \n'  
               'Train Acc:  {:.4f}, Test Acc:  {:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f} \n').format(
                   info['current_epoch'], info['epochs'], info['t_duration'],
                   info['train_loss'], info['test_loss'][0], info['test_loss'][1], info['test_loss'][2], info['test_loss'][3], info['test_loss'][4], info['test_loss'][5], info['test_loss'][6], info['test_loss'][7], 
                   info['train_acc'], info['test_acc'][0], info['test_acc'][1], info['test_acc'][2], info['test_acc'][3], info['test_acc'][4], info['test_acc'][5], info['test_acc'][6], info['test_acc'][7])
    print(message)


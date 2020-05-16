import ann.ann_runnable 
import ann.ann_menu
import sys

if __name__ == '__main__':
    """ann.ann_runnable.start(
    sys.argv[1], 
           [8,32,64],
           [   
               'I131_treatment',
               'goitre', 'lithium', 'hypopituitary'
           ]
           ,  0.1) 
    """
    perc, matrix = ann.ann_runnable.launchConfusion( "test", 64, 0.9, [  
               'TSH', 'FTI', 'TT4', 'T3',
               'T4U', 'age', 'on_thyroxine'
               ,'referral_source','pregnant',
               'sex', 'tumor', 'query_hyperthyroid' ]) 
    ann.ann_runnable.printConfusionMatrix(matrix)
    ann.ann_network.LOG( str(perc) )

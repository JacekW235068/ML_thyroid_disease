import ann.ann_runnable 
import ann.ann_menu
if __name__ == '__main__':
    ann.ann_runnable.start(
            "output", 
            [8],
            [   
                'TSH', 'FTI', 'TT4', 'T3',
                'T4U', 'age', 'on_thyroxine'
                ,'referral_source','pregnant',
                'sex', 'tumor', 'query_hyperthyroid', 
                'query_hypothyroid', 'thyroid_surgery', 
                'psych', 'sick', 'query_on_thyroxine', 
                'on_antithyroid_medication', 'I131_treatment', 
                'goitre', 'lithium', 'hypopituitary'

            ]
            ,  0.1)

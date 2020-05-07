from typing import NamedTuple 
import configparser 


glbListNeurons = [10]
glbListFeatures = ['none']

def menu_settings_load():
    toRetFeat = -1
    toRetNeur = -1

    cfgParser = configparser.RawConfigParser()
    cfgParser.read('config.cfg')
    try:
        listNeurons = cfgParser.get('SETTINGS', 'NEURONS').split(',')
        if( len(listNeurons) < 1 ):
            print("Wadliwy format kolejnych liczb neuronow")
        else:
            toRetNeur = listNeurons

        listFeature = cfgParser.get('SETTINGS', 'FEATURES').split(',')
        if( len(listNeurons) < 1 ):
            print("Wadliwy format kolejnych liczb neuronow")
        else:
            toRetFeat = listFeature 
        
    except configparser.NoSectionError:
        print("Nie odnaleziono poprawnej sekcji. \
                Upewnij sie ze istnieje plik config.cfg i posiada sekcje SETTINGS z polem NEURONS i FEATURES.")

    return toRetFeat, toRetNeur

def print_settings_info(feat,stage):
    print("-------- NEURONY ------")
    print("Ilość warstw:", len(stage))
    print("Neurony: ") 
    for element  in stage:
        print( element )
    print("------- CECHY --------")
    for feature in feat:
        print( feature )

    print("-----------------------")
def menu_settings_print(bReadOnly):
    print("Edycja ustawien badan")
    if(bReadOnly):
        print("Tryb odczytu")
    else:
        print("Tryb edycji") 
    
    print('list \t\t- wylistuj wszystkie załadowane etapy')
    print('toggle \t\t- przełącz tryb') 
    print("b \t\t- powrót")

    if(not bReadOnly):
        print('load \t\t - załaduj etapy z pliku')
    print("------------------------------")

def menu_settings():
    global glbListNeurons
    global glbListFeatures 

    bReadOnly = True 
    szResponse = " "
    while( szResponse[0] != 'b'):
        menu_settings_print(bReadOnly)
        szResponse = input(">")
        if(szResponse == 'list'):
            print_settings_info(glbListFeatures, glbListNeurons)
        elif( szResponse == 'toggle' ):
            bReadOnly = not bReadOnly 
        elif( szResponse == 'load' ):
            if( not bReadOnly):
                retFeat, retVal = menu_settings_load()
                if retVal != -1 and retFeat != -1:
                    glbListFeatures = retFeat
                    glbListNeurons = retVal 
    return

def menu_start():
    global glbListNeurons 
    print_settings_info(glbListFeatures, glbListNeurons)
    print("Czy są to poprawne dane do uruchomienia badań?")
    cAnsw = input('y/n:')
    if(cAnsw[0] == 'y'):
        print("GO")
    return 
def menu_open():
    while( True ):
        print("config\t- konfiguracja badan")
        print("start\t- uruchom badania")
        szAnsw = input('>')    
        if szAnsw == 'config':
            menu_settings() 
        elif szAnsw == 'start':
            menu_start()


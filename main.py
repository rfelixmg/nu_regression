# Carrega um arquivo csv em uma array
def load_csv_file(objName, fileNamePath, delim=';'):
    """       
         
        load_csv_file:
            Retorna um array de um arquivo CSV
        Parametros
        ----------
        objName: string
            Nome do arquivo csv
        fileNamePath: string
            Diretorio local onde carregar o arquivo
        delim: string
            Caractere que delimita cada coluna
         
        Examples
        --------
        >>> import file_utils
        >>> file_utils.load_csv_file('teste', ';')    # Arquivo a ser carregado
        >>> [1,2,3]
         
    """
    import csv
    import numpy as np
 
    csvfile = open(fileNamePath + objName + '.csv', 'rb')
    reader = csv.reader(csvfile, delimiter=delim)
    cdata = list(reader)
    array_from_file = np.array(cdata)
    csvfile.close()
 
    return array_from_file
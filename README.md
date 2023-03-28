# GPT-2 per il planning

Repository contenente il codice realizzato per effettuare il preprocessing di un dataset e il training/testing di un modello (GPT-2).
Il contenuto delle cartelle è il seguente:

* config_files: file JSON con le opzioni per eseguire i vari script
* dataset: contiene il dataset utilizzato per l'addestramento e per i test
* dict_pickles: ogni file pickle in questa cartella contiene un dizionario Python con la lista di tutti i fatti/azioni possibili per i problemi per il dominio considerato
* notebooks: alcuni notebook Jupyter con il codice utilizzato per creare i tokenizer e altro
* pddl: file dei problemi utilizzati in formato PDDL
* plan_info: informazioni di base sui piani del dataset
* plan_utils: alcune funzioni di utilità scritte in Python
* scripts: tutti gli script Python utilizzati per il preprocessing del dataset, l'addestramento e il testing del modello
* tokenizers: contiene i tokenizer necessari per il funzionamento del modello

## Requisiti

La versione di Python utilizzata è la 3.9.13, mentre quelle che seguono sono le principali librerie utilizzate:

* pytorch = 1.12.1
* transformers = 4.24.0
* tokenizers = 0.11.4
* accelerate = 0.15.0
* datasets = 2.6.1
* evaluate = 0.3.0
* scikit-learn = 1.1.3
* numpy = 1.23.4

Per addestrare il modello utilizzando GPU è necessario aver installato le librerie richieste da PyTorch, nel caso di GPU NVIDIA è richiesta l'installazione di CUDA Toolkit. Questa installazione può essere fatta tramite conda.

* cudatoolkit = 11.4.4

## Utilizzare gli script

Per addestrare il modello utilizzando una o più GPU è possibile utilizzare accelerate. Con il comando `accelerate config` è possibile indicare quali GPU utilizzare, eventuali librerie da utilizzare durante l'addestramento ed altre opzioni. Una volta fatto ciò per addestrare il modello basta fare:

```
accelerate launch scripts/training_script.py config_files/training_script_config.json
```

La maggior parte degli script realizzati accetta delle opzioni da linea di comando; tali opzioni sono riportate in una dataclass all'inizio dello script oppure è possibile visualizzarle utilizzando `python <nome_script>.py -h`. Le opzioni possono essere fornite anche utilizzando un file JSON, come nell'esempio mostrato:

```json
{
    "dataset_dir": "plans",
    "tokenizer_file": "logistics_tokenizer.json",
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 16,
    "num_train_epochs": 40,
    "output_dir": "training_gpt",
    "seed": 7,
    "checkpointing_steps": "epoch",
    "save_total_limit": 5
}
```
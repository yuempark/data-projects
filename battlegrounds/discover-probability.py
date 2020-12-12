import pandas as pd
import numpy as np
import tkinter as tk

counts = pd.read_csv('counts.csv',index_col=0)

# set up the window
window = tk.Tk()
window.title('discover probability')
window.geometry('1000x300')

# removed 1
removed1_value = tk.StringVar()
def get_removed1(event):
    removed1_value.set(removed1_listbox.get('anchor'))
removed1_label = tk.Label(master=window, text='type removed - 1')
removed1_listbox = tk.Listbox(master=window, exportselection=False)
for item in counts.index[:-1]:
    removed1_listbox.insert('end', item)
removed1_listbox.bind('<<ListboxSelect>>',get_removed1)

# removed 2
removed2_value = tk.StringVar()
def get_removed2(event):
    removed2_value.set(removed2_listbox.get('anchor'))
removed2_label = tk.Label(master=window, text='type removed - 2')
removed2_listbox = tk.Listbox(master=window, exportselection=False)
for item in counts.index[:-1]:
    removed2_listbox.insert('end', item)
removed2_listbox.bind('<<ListboxSelect>>',get_removed2)

# tier
tier_value = tk.StringVar()
def get_tier(event):
    tier_value.set(tier_listbox.get('anchor'))
tier_label = tk.Label(master=window, text='tier')
tier_listbox = tk.Listbox(master=window, exportselection=False)
for item in counts.columns:
    tier_listbox.insert('end', item)
tier_listbox.bind('<<ListboxSelect>>',get_tier)

# number
def btn_increase():
    num_value = int(num_value_label['text'])
    num_value_label['text'] = str(int(num_value)+1)
def btn_decrease():
    num_value = int(num_value_label['text'])
    num_value_label['text'] = str(int(num_value)-1)
num_label = tk.Label(master=window, text='number')
num_decrease = tk.Button(master=window, text='-', command=btn_decrease)
num_value_label = tk.Label(master=window, text='1')
num_increase = tk.Button(master=window, text='+', command=btn_increase)

# output probability
def calculate_probability():
    
    # get the number of total minions in the pool
    n_total_tier = int(np.sum(counts[tier_value.get()]))
    n_removed1_tier = int(counts[tier_value.get()][removed1_value.get()])
    n_removed2_tier = int(counts[tier_value.get()][removed2_value.get()])
    n = n_total_tier - n_removed1_tier - n_removed2_tier
    
    # get the number we can choose from
    k = int(num_value_label['text'])
    
    # case of 1
    p1 = ((k*(n-k)*(n-k-1)) / (n*(n-1)*(n-2))) * 3
    
    # case of 2
    if k>=2:
        p2 = ((k*(k-1)*(n-k)) / (n*(n-1)*(n-2))) * 3
    else:
        p2 = 0
        
    # case of 3
    if k>=3:
        p3 = ((k*(k-1)*(k-2)) / (n*(n-1)*(n-2)))
    else:
        p3 = 0
        
    probability = p1 + p2 + p3
    
    probability_value_label['text'] = str(np.round(probability*100,1))+ ' %'

probability_button = tk.Button(master=window, text='calculate', command=calculate_probability)
probability_value_label = tk.Label(master=window, text='')
    
# show the widgets
removed1_label.grid(row=0, column=0, padx=10)
removed1_listbox.grid(row=1, column=0, padx=10)
removed2_label.grid(row=0, column=1, padx=10)
removed2_listbox.grid(row=1, column=1, padx=10)
tier_label.grid(row=0, column=2, padx=10)
tier_listbox.grid(row=1, column=2, padx=10)
num_label.grid(row=0, column=4, padx=10)
num_decrease.grid(row=1, column=3, padx=10)
num_value_label.grid(row=1, column=4, padx=10)
num_increase.grid(row=1, column=5, padx=10)
probability_button.grid(row=0, column=6, padx=10)
probability_value_label.grid(row=1, column=6, padx=10)

window.mainloop()
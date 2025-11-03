import torch

reflow_data1 = torch.load('G4_reflow_DATASET_50.pth', map_location='cpu')
reflow_data2 = torch.load('G4_reflow_DATASET_50_to_100.pth', map_location='cpu')
reflow_data= []
reflow_data.append(torch.cat([reflow_data1[0], reflow_data2[0]], dim=0))
reflow_data.append(torch.cat([reflow_data1[1], reflow_data2[1]], dim=0))
torch.save(reflow_data, 'G4_reflow_DATASET_100.pth' )
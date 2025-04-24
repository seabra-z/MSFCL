import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
from PIL import Image


model_path = "data/ddinter/drug_emb_trimnet4.npy"
drug_embeddings = np.load(model_path, allow_pickle=True)


def visualize_drug(smiles, drug_index, title):
    mol = Chem.MolFromSmiles(smiles)

    drug_embedding = drug_embeddings[drug_index]

    num_atoms = mol.GetNumAtoms()
    atom_attention_weights = drug_embedding[:num_atoms]

    max_weight_atom = int(np.argmax(atom_attention_weights))

    neighbors = [n.GetIdx() for n in mol.GetAtomWithIdx(max_weight_atom).GetNeighbors()]

    highlight_atoms = [max_weight_atom] + neighbors
    highlight_colors = {i: (1.0, 0.0, 0.0) for i in highlight_atoms}

    highlight_bonds = []
    for bond in mol.GetBonds():
        begin_atom = bond.GetBeginAtomIdx()
        end_atom = bond.GetEndAtomIdx()
        if begin_atom in highlight_atoms and end_atom in highlight_atoms:
            highlight_bonds.append(bond.GetIdx())

    img = Draw.MolToImage(
        mol,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=highlight_colors,
        highlightBonds=highlight_bonds,
        highlightBondColors={i: (1.0, 0.0, 0.0) for i in highlight_bonds},
        size=(400, 400),
        legend=title
    )
    return img


alosetron_smiles = "CC(C)NCC(O)COC1=CC=C(CC(N)=O)C=C1"
alosetron_img = visualize_drug(alosetron_smiles, 98, "Atenolol")  


apomorphine_smiles = "CCOC(=O)C1=C(COCCN)NC(C)=C(C1C1=CC=CC=C1Cl)C(=O)OC"
apomorphine_img = visualize_drug(apomorphine_smiles, 61, "Amlodipine") 


max_height = max(alosetron_img.size[1], apomorphine_img.size[1])
total_width = alosetron_img.size[0] + apomorphine_img.size[0]


combined_img = Image.new('RGB', (total_width, max_height), 'white')
combined_img.paste(alosetron_img, (0, 0))
combined_img.paste(apomorphine_img, (alosetron_img.size[0], 0))


combined_img.save("Atenolol_Amlodipine_visualization.png")

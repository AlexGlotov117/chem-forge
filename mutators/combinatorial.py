from rdkit import Chem
import random
# import selfies as sf
from rdkit.Chem import rdChemReactions

# Standard database-heavy anions commonly found in thermo/chemicals tables
COMMON_DATABASE_ANIONS = [
    "[BH4-]",          # Borohydride
    "[Cl-]",           # Chloride
    "[Br-]",           # Bromide
    "[I-]",            # Iodide
    "F[B-](F)(F)F",    # Tetrafluoroborate (BF4)
    "F[P-](F)(F)(F)(F)F", # Hexafluorophosphate (PF6)
    "[N-]([S](=O)(=O)C(F)(F)F)[S](=O)(=O)C(F)(F)F", # NTf2 (Bis(trifluoromethanesulfonyl)imide)
    "[O-]S(=O)(=O)C(F)(F)F", # Triflate (OTf)
    "[O-]C(=O)C",      # Acetate
    "[NO3-]",          # Nitrate
]

def mutator(target_smiles, num_mutations_per_target=100):
    # try:
    #     # 1. Convert SMILES to SELFIES
    #     target_selfies = sf.encoder(target_smiles)
    #     tokens = list(sf.split_selfies(target_selfies))
    # except Exception:
    #     return

    # # A pool of common organic & ionic tokens to insert/substitute
    # token_pool = [
    #     "[C]", "[C][C]", "[Branch1]", "[Ring1]", 
    #     "[N+1]", "[O]", "[F]", "[Cl]", "[Br]", "[Expl=B-1]"
    # ]

    # for _ in range(num_mutations_per_target):
    #     mutated_tokens = tokens.copy()
    #     mutation_type = random.choice(["substitute", "insert", "delete"])
    #     idx = random.randint(0, len(mutated_tokens) - 1)

    #     if mutation_type == "substitute" and len(mutated_tokens) > 0:
    #         mutated_tokens[idx] = random.choice(token_pool)
            
    #     elif mutation_type == "insert":
    #         mutated_tokens.insert(idx, random.choice(token_pool))
            
    #     elif mutation_type == "delete" and len(mutated_tokens) > 1:
    #         mutated_tokens.pop(idx)

    #     # Reconstruct SELFIES and decode back to SMILES
    #     try:
    #         mutated_selfies = "".join(mutated_tokens)
    #         candidate_smiles = sf.decoder(mutated_selfies)
            
    #         # Canonicalize and validate via RDKit
    #         mol = Chem.MolFromSmiles(candidate_smiles)
    #         if mol is None:
    #             continue
                
    #         canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
            
    #         # Skip returning exact target copy
    #         if canonical_smiles != Chem.MolToSmiles(Chem.MolFromSmiles(target_smiles), canonical=True):
    #             yield canonical_smiles
                
    #     except Exception:
    #         continue
    mol = Chem.MolFromSmiles(target_smiles)
    if mol is None:
        return

    # 1. Deconstruct Target into Cation and Anion
    frags = Chem.GetMolFrags(mol, asMols=True)
    cation_mols = []
    
    if len(frags) >= 2:
        for f in frags:
            f_smiles = Chem.MolToSmiles(f, canonical=True)
            # Separate organic/cation fragment from simple counterions
            if "+" in f_smiles or not ("B" in f_smiles or "-" in f_smiles):
                cation_mols.append(f)
    if not cation_mols:
        cation_mols = [mol]

    # 2. Comprehensive Cation Transformation SMARTS
    cation_reactions = [
        # Alkyl Extensions (C1 -> C2 -> C3 -> C4)
        '[C;H3:1] >> [C:1]C',                   # Add Methyl
        '[C;H3:1] >> [C:1]CC',                  # Add Ethyl
        '[C;H3:1] >> [C:1]CCC',                 # Add Propyl
        '[C;H3:1] >> [C:1]CCCC',                # Add Butyl
        
        # Branching & Isomerization
        '[C;H2:1] >> [CH:1](C)C',               # Isopropyl branching
        
        # Nitrogen / Carbon Functionalization (Alcohol / Hydroxyl ILs like Choline)
        '[C;H3:1] >> [C:1]CO',                  # Ethanolamine derivative
        
        # Symmetrical Alkyl Extension on Quaternary Nitrogen
        '[N+:1](C)(C)C >> [N+:1](CC)(CC)CC',    # TMA -> TEA
        '[N+:1](CC)(CC)CC >> [N+:1](CCC)(CCC)CCC', # TEA -> TPA
        '[N+:1](CCC)(CCC)CCC >> [N+:1](CCCC)(CCCC)CCCC', # TPA -> TBA
    ]

    compiled_rxns = [rdChemReactions.ReactionFromSmarts(s) for s in cation_reactions]

    # 3. Generate Mutated Cations
    generated_cations = set()
    
    for cat_mol in cation_mols:
        # Include original cation
        generated_cations.add(Chem.MolToSmiles(cat_mol, canonical=True))
        
        # Apply SMARTS Transformations
        for rxn in compiled_rxns:
            try:
                products = rxn.RunReactants((cat_mol,))
                for prod_tuple in products:
                    for prod in prod_tuple:
                        try:
                            Chem.SanitizeMol(prod)
                            generated_cations.add(Chem.MolToSmiles(prod, canonical=True))
                        except Exception:
                            continue
            except Exception:
                continue

    # 4. Combinatorial Cross-Product: (Mutated Cations) x (Common DB Anions)
    for cat_smiles in generated_cations:
        # A. Yield pure cation alone (if cataloged)
        yield cat_smiles
        
        # B. Yield cation paired with every common database anion
        for anion_smiles in COMMON_DATABASE_ANIONS:
            full_salt = f"{cat_smiles}.{anion_smiles}"
            validated_mol = Chem.MolFromSmiles(full_salt)
            if validated_mol is not None:
                yield Chem.MolToSmiles(validated_mol, canonical=True)
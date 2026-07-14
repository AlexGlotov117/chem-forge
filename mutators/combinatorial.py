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

import selfies as sf
import random

def selfies_mutator(smiles, num_mutations = 3):
    """Generates a valid mutant without any hardcoded SMARTS rules."""
    selfies_str = sf.encoder(smiles)
    tokens = list(sf.split_selfies(selfies_str))
    
    # Perform random mutation (replacement/insertion/deletion)
    alphabet = list(sf.get_semantic_robust_alphabet())
    for _ in range(num_mutations):
        idx = random.randint(0, len(tokens) - 1)
        tokens[idx] = random.choice(alphabet)
        
    mutated_selfies = "".join(tokens)
    return sf.decoder(mutated_selfies)

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

class SmartActionMutator:
    """
    A modular, action-based mutation engine. 
    New chemical transformations can be registered dynamically as isolated actions.
    """
    def __init__(self):
        self._actions = {}
        self.history = []

    def register_action(self, name, action_fn):
        """Register a new mutation action into the engine."""
        self._actions[name] = action_fn

    def list_actions(self):
        """List all currently registered mutation actions."""
        return list(self._actions.keys())
    
    def clear_history(self):
        """Reset the mutation history log."""
        self.history = []

    def mutate(self, target_smiles, depth=3, active_actions=None):
        """
        Mutates target SMILES up to `depth` steps.
        """
        self.clear_history()
        
        current_frontier = {target_smiles}
        visited = {target_smiles}

        for current_depth in range(depth):
            next_frontier = set()
            
            for parent_smi in current_frontier:
                mol = Chem.MolFromSmiles(parent_smi)
                if mol is None:
                    continue
                    
                actions_to_run = active_actions or self._actions.keys()
                for action_name in actions_to_run:
                    action_fn = self._actions[action_name]
                    
                    for candidate_smi in action_fn(mol):
                        if candidate_smi in visited:
                            self.history.append({
                                "action": action_name,
                                "candidate": candidate_smi,
                                "status": "SKIPPED_DUPLICATE"
                            })
                            continue

                        visited.add(candidate_smi)
                        next_frontier.add(candidate_smi)
                        
                        self.history.append({
                            "action": f"{action_name} (depth {current_depth+1})",
                            "candidate": candidate_smi,
                            "status": "GENERATED"
                        })
                        yield candidate_smi

            current_frontier = next_frontier

    def mutate_until_k(self, target_smiles: str, k_neighbors: int = 10, max_depth: int = 10):
        """
        Iteratively increases mutation depth (d = 1, 2, 3...) until
        at least k_neighbors unique valid candidate SMILES are generated.
        """
        from collections import deque
        visited = {target_smiles}
        candidates = []
        
        # Queue stores tuples of (smiles, current_depth)
        queue = deque([(target_smiles, 0)])

        while queue and len(candidates) < k_neighbors:
            current_smi, current_depth = queue.popleft()

            # If we've reached max depth limit and already have candidates, stop
            if current_depth >= max_depth:
                break

            mol = Chem.MolFromSmiles(current_smi)
            if mol is None:
                continue

            # Run all registered mutation actions at the current depth level
            for action_name, action_fn in self._actions.items():
                for mutant_smi in action_fn(mol):
                    if mutant_smi not in visited:
                        visited.add(mutant_smi)
                        candidates.append(mutant_smi)
                        
                        # Queue the new mutant for the next depth layer (complexity increase)
                        queue.append((mutant_smi, current_depth + 1))
                        
                        self.history.append({
                            "action": f"{action_name} (depth {current_depth + 1})",
                            "candidate": mutant_smi,
                            "status": "GENERATED"
                        })

                        # Early exit if we reached target neighbor count
                        if len(candidates) >= k_neighbors:
                            return candidates

        return candidates
    
    def print_mutation_report(self):
        """Prints a clean summary table of all tried mutations."""
        print("\n" + "="*70)
        print("                 MUTATION MUTATOR ATTEMPT REPORT               ")
        print("="*70)
        print(f"{'#':<4} | {'Action':<20} | {'Status':<25} | {'Candidate SMILES'}")
        print("-" * 70)
        for i, entry in enumerate(self.history, 1):
            print(f"{i:<4} | {entry['action']:<20} | {entry['status']:<25} | {entry['candidate']}")
        print("="*70 + "\n")


# =====================================================================
# INDIVIDUAL ATOMIC ACTIONS (Add or remove actions here easily!)
# =====================================================================

def action_halogen_swap(mol):
    """Action: Swaps Cl <-> Br <-> F <-> I on aliphatic or aromatic sites."""
    rules = [
        '[Cl:1] >> [Br:1]', '[Br:1] >> [Cl:1]',
        '[Cl:1] >> [F:1]',  '[F:1] >> [Cl:1]',
        '[Cl:1] >> [I:1]',  '[I:1] >> [Cl:1]'
    ]
    for smarts in rules:
        rxn = rdChemReactions.ReactionFromSmarts(smarts)
        products = rxn.RunReactants((mol,))
        for prod_tuple in products:
            for prod in prod_tuple:
                try:
                    Chem.SanitizeMol(prod)
                    yield Chem.MolToSmiles(prod, canonical=True)
                except Exception:
                    continue


def action_alkyl_extension(mol):
    """Action: Adds a methyl (-CH3) group to terminal carbons."""
    rxn = rdChemReactions.ReactionFromSmarts('[C;H3:1] >> [C:1]C')
    products = rxn.RunReactants((mol,))
    for prod_tuple in products:
        for prod in prod_tuple:
            try:
                Chem.SanitizeMol(prod)
                yield Chem.MolToSmiles(prod, canonical=True)
            except Exception:
                continue


def action_counterion_swap(mol):
    """Action: Swaps cations/anions if the target is an ionic salt."""

    # Standard database salts
    ANIONS = ["[Cl-]", "[Br-]", "[I-]", "F[P-](F)(F)(F)(F)F", "F[B-](F)(F)F", "[NO3-]"]
    CATIONS = ["[Li+]", "[Na+]", "[K+]", "[NH4+]", "CC[N+](CC)(CC)CC"]

    # Break target into separate molecular fragments
    frags = Chem.GetMolFrags(mol, asMols=True)
    
    for frag in frags:
        frag_smi = Chem.MolToSmiles(frag, canonical=True)
        
        # If fragment is a CATION (+1), pair it 1:1 with database ANIONS (-1)
        if "+" in frag_smi and "-" not in frag_smi:
            for anion in ANIONS:
                salt_smi = f"{frag_smi}.{anion}"
                salt_mol = Chem.MolFromSmiles(salt_smi)
                if salt_mol:
                    yield Chem.MolToSmiles(salt_mol, canonical=True)
                    
        # If fragment is an ANION (-1), pair it 1:1 with database CATIONS (+1)
        elif "-" in frag_smi and "+" not in frag_smi:
            for cation in CATIONS:
                salt_smi = f"{frag_smi}.{cation}"
                salt_mol = Chem.MolFromSmiles(salt_smi)
                if salt_mol:
                    yield Chem.MolToSmiles(salt_mol, canonical=True)

def action_alcohol_to_ether(mol):
    """Converts -OH to -OCH3 (e.g., CCO -> CCOC)."""
    rxn = rdChemReactions.ReactionFromSmarts('[OH:1] >> [OCH3:1]')
    for prod_tuple in rxn.RunReactants((mol,)):
        for prod in prod_tuple:
            try:
                Chem.SanitizeMol(prod)
                yield Chem.MolToSmiles(prod, canonical=True)
            except Exception:
                continue

def action_hydroxyl_to_halogen(mol):
    """Replaces -OH with -Cl or -Br (e.g., CCO -> CCCl, CCBr)."""
    rules = ['[OH:1] >> [Cl:1]', '[OH:1] >> [Br:1]']
    for smarts in rules:
        rxn = rdChemReactions.ReactionFromSmarts(smarts)
        for prod_tuple in rxn.RunReactants((mol,)):
            for prod in prod_tuple:
                try:
                    Chem.SanitizeMol(prod)
                    yield Chem.MolToSmiles(prod, canonical=True)
                except Exception:
                    continue

def action_chain_branching(mol):
    """Converts linear methylene into isopropyl branching: -CH2- -> -CH(CH3)-"""
    rxn = rdChemReactions.ReactionFromSmarts('[C;H2:1] >> [CH:1](C)C')
    for prod_tuple in rxn.RunReactants((mol,)):
        for prod in prod_tuple:
            try:
                Chem.SanitizeMol(prod)
                yield Chem.MolToSmiles(prod, canonical=True)
            except Exception:
                continue

def action_n_alkylation(mol):
    """
    Action: Adds methyl groups to Nitrogen atoms bound to Boron or Hydrogen.
    Matches: [NH3+], [NH2+], [NH+], or neutral amines.
    """
    rules = [
        '[N;H3:1] >> [N:1]C',
        '[N;H2:1] >> [N:1]C',
        '[N;H1:1] >> [N:1]C'
    ]
    for smarts in rules:
        rxn = rdChemReactions.ReactionFromSmarts(smarts)
        for prod_tuple in rxn.RunReactants((mol,)):
            for prod in prod_tuple:
                try:
                    Chem.SanitizeMol(prod)
                    yield Chem.MolToSmiles(prod, canonical=True)
                except Exception:
                    continue


def action_boron_halogenation(mol):
    """
    Action: Replaces H on Boron with Halogens (F, Cl).
    Converts [BH3-] -> [BH2-](F), [BH2-](Cl), etc.
    """
    rules = [
        '[B;H3:1] >> [B:1]F', '[B;H3:1] >> [B:1]Cl',
        '[B;H2:1] >> [B:1]F', '[B;H2:1] >> [B:1]Cl',
        '[B;H1:1] >> [B:1]F', '[B;H1:1] >> [B:1]Cl'
    ]
    for smarts in rules:
        rxn = rdChemReactions.ReactionFromSmarts(smarts)
        for prod_tuple in rxn.RunReactants((mol,)):
            for prod in prod_tuple:
                try:
                    Chem.SanitizeMol(prod)
                    yield Chem.MolToSmiles(prod, canonical=True)
                except Exception:
                    continue


def action_b_alkylation(mol):
    """
    Action: Replaces H on Boron with Methyl groups.
    Converts [BH3-] -> [BH2-](C).
    """
    rules = [
        '[B;H3:1] >> [B:1]C',
        '[B;H2:1] >> [B:1]C'
    ]
    for smarts in rules:
        rxn = rdChemReactions.ReactionFromSmarts(smarts)
        for prod_tuple in rxn.RunReactants((mol,)):
            for prod in prod_tuple:
                try:
                    Chem.SanitizeMol(prod)
                    yield Chem.MolToSmiles(prod, canonical=True)
                except Exception:
                    continue

# =====================================================================
# DEFAULT FACTORY FUNCTION
# =====================================================================

def create_smart_mutator():
    """Builds and registers standard action set."""
    engine = SmartActionMutator()
    
    engine.register_action("halogen_swap", action_halogen_swap)
    engine.register_action("alkyl_extension", action_alkyl_extension)
    engine.register_action("counterion_swap", action_counterion_swap)
    engine.register_action("alcohol_to_ether", action_alcohol_to_ether)
    engine.register_action("hydroxyl_to_halogen", action_hydroxyl_to_halogen)
    engine.register_action("chain_branching", action_chain_branching)
    engine.register_action("n_alkylation", action_n_alkylation)
    engine.register_action("boron_halogenation", action_boron_halogenation)
    engine.register_action("b_alkylation", action_b_alkylation)

    return engine
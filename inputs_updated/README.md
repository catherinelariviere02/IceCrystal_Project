## Notes on Shapes 

Generated using new crystal2shape procedure (using hierarchy and basis set) 

Each approach (i.e. tolerances, cutoff distances) are noted in subfolders for shapes. 

148 - Ice II (no disordered counterpart)
141 - Ice VIII (ordered ice VI)
92 - Ice IX (ordered ice III)
36 - Ice XI (ordered ice Ih)
14 - Ice XIII (ordered ice V)
19 - Ice XIV (ordered ice XII)
59 - Ice XV (ordered ice VI)
81 - Ice XIX (ordered ice VI)

Notes on shape generation: 

- 14: ice XIII - each only has one wyckoff site (C1 symmetry), there are 7 equivalent sites for oxygen, 14 equivalent sites for hydrogen
    - most of the equivalent sites for oxygen produce slightly different irregular truncated tetrahedra (only is significantly different) 
    - all take required hierarchy length of 2 (wondering if that's true for all oxygen give hydrogen network constraints?)
    - hydrogen equiv sites: 756, 864, 972, 1080, 1188, 1296, 1404, 1512, 1620, 1728, 1836, 1944, 2052, 2160 



Notes on unit cell generation: 

- SO: there are, to my mind, "correct" orientations of the oxygen particles, which keep the hydrogen the correct distances from their centroids. This is what could guarantee that the hydrogen order is maintained in a hard particle system --> the order is determined by the distance of the hydrogen away 
    - I would guess that this is why creating the unit cell without moving the particles is hard: if the oxygen are in the "wrong" orientation when they get within contact distance of the hydrogen, then they will get stuck there are they won't have anywhere to go   
    - this is also what will make self-assembly of these particular structures extremely difficult 
    - This is also what makes the approach 
    - The features of the first level of BOD shape hierarchy are simple enough to maintain in ice VIII that I think it makes sense 
    - what I'm finding is that (seemingly randomly) the unit cells reach a point at which they cannot reorient out of the jammed state they're in when I don't let them move 

    Q: do the oxygen tend to find the "correct" orientations with the translation springs? 
    - yes for ice VIII (upon visual inspection)
    - for ice IX: 
        - what is the "close" and "far" sides for ice IX? 
    
    Generally: 
    - how do you characterize the "close" and "far" side: need to choose the principle axis of symmetry, and find the distance between each of the 4 major faces 
    - do we need to have the 4 major faces? I think so contact 
    - 
 

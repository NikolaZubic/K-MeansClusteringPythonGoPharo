iter := 22.
mode := 'sequential'.

filePath := 'C:\Users\Nikola Zubic\Desktop\NTP projekat\K-MeansClusteringPythonGoPharo\Python\results\secondExample\', mode ,'\current_state_', (iter printString ) ,'.txt'.

string := FileStream readOnlyFileNamed: filePath do: [ :stream | stream contents ].

splitted := string splitOn: '|'.

map := splitted collect: [ :line | (line splitOn: Character space)
									collect: [ :number | number asNumber ]
								].
size := map size.

points :=  (1 to: size) collect: [ :i | ((map at: i) at: 1) @ ((map at: i) at: 2) ].

firstCluster := Set new.
secondCluster := Set new.
thirdCluster := Set new.
fourthCluster := Set new.
centroids := Set new.

map do: [ :each |
	(((each) at: 3) = 1)
	ifTrue: [centroids add: ((each) at: 1) @ ((each) at: 2)]
	].

map do: [ :each |
	(((each) at: 4) = 0)
	ifTrue: [firstCluster add: ((each) at: 1) @ ((each) at: 2)]
	].

map do: [ :each |
	(((each) at: 4) = 1)
	ifTrue: [secondCluster add: ((each) at: 1) @ ((each) at: 2)]
	].

map do: [ :each |
	(((each) at: 4) = 2)
	ifTrue: [thirdCluster add: ((each) at: 1) @ ((each) at: 2)]
	].

map do: [ :each |
	(((each) at: 4) = 3)
	ifTrue: [fourthCluster add: ((each) at: 1) @ ((each) at: 2)]
	].

cluster1ds := RTData new.
cluster1ds dotShape color: Color red.
cluster1ds points: firstCluster; x: #x; y: #y.

cluster2ds := RTData new.
cluster2ds dotShape color: Color blue.
cluster2ds points: secondCluster; x: #x; y: #y.


cluster3ds := RTData new .
cluster3ds dotShape color: Color green.
cluster3ds points: thirdCluster; x: #x; y: #y.

cluster4ds := RTData new .
cluster4ds dotShape color: Color yellow.
cluster4ds points: fourthCluster; x: #x; y: #y.

centroidsRT := RTData new .
centroidsRT dotShape color: Color white.
centroidsRT dotShape size: 8.
centroidsRT points: centroids; x: #x; y: #y.

graphed := RTGrapher new
    add: cluster1ds;
    add: cluster2ds;
    add: cluster3ds;
    add: cluster4ds;
    add: centroidsRT;
    extent: 500@500;
    "maxX: 5; maxY: 5;"
    build;
    view.
.

graphed view.

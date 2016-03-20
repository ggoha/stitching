var scena = {};
scena.camerasId = 0; scena.linesId = 0; scena.pointsId = 0;
var MAXX = window.innerWidth, MAXY = window.innerHeight; VALIDY = 700;
scena.H = VALIDY;
var matching = [], points = [];
var textureLine = [], texturePoint= [];
textureLine.push(PIXI.Texture.fromImage('2D/line0.png'));
textureLine.push(PIXI.Texture.fromImage('2D/line1.png'));
textureLine.push(PIXI.Texture.fromImage('2D/line2.png'));
textureLine.push(PIXI.Texture.fromImage('2D/line3.png'));
textureLine.push(PIXI.Texture.fromImage('2D/line4.png'));
texturePoint.push(PIXI.Texture.fromImage('2D/point0.png'));
texturePoint.push(PIXI.Texture.fromImage('2D/point1.png'));
texturePoint.push(PIXI.Texture.fromImage('2D/point2.png'));
texturePoint.push(PIXI.Texture.fromImage('2D/point3.png'));
texturePoint.push(PIXI.Texture.fromImage('2D/point4.png'));

scena.addCamera = function ()
{
    var cameraContainer = new PIXI.Container();
    cameras.addChild(cameraContainer);

    var camera = new PIXI.Graphics();
    camera.position.x = Math.floor(Math.random() * MAXX);
    camera.position.y = Math.floor(Math.random() * VALIDY);

    camera.beginFill(0x000000);
     
    // draw a rectangle
    camera.drawCircle(0, 0, 2);
    camera.endFill();
    camera.beginFill(0xF0F0F0);
    camera.drawRect(-30, -30, 60, 1);
    camera.endFill();

    // enable the bunny to be interactive... this will allow it to respond to mouse and touch events
    camera.interactive = true;
    camera.hitArea = new PIXI.Circle(0, 0, 10);

    // this button mode will mean the hand cursor appears when you roll over the bunny with your mouse
    camera.buttonMode = true;  

    // make it a bit bigger, so it's easier to grab
    camera.scale.set(3);

    // setup events
    camera
        // events for drag start
        .on('mousedown', onDragStart)
        // events for drag end
        .on('mouseup', onDragEnd)
        .on('mouseupoutside', onDragEnd)
        // events for drag move
        .on('mousemove', onDragMove);

    camera.isMatching = false;
    camera.autoRender = true;

    // add it the stage so we see it on our screens..
    cameraContainer.addChild(camera);

    var projection = new PIXI.Container();
    cameraContainer.addChild(projection);

    var match = new PIXI.Container();
    cameraContainer.addChild(match);

    cameraContainer.project = function ()
    {
        //меняем порядок слоев, для их правильного отображения на на проекции
        for (x in lines.children)
            for (y in lines.children)
                if (x > y && lines.children[x].y > lines.children[y].y)
                    lines.swapChildren(lines.children[x], lines.children[y]);
        //очищаем будущую проекцию
        this.children[1].removeChildren()
        for (x in lines.children)
        {
            //находим координаты концов слоев и камеры
            var x1 = lines.children[x].position.x, y1 = lines.children[x].position.y;
            var x2 = x1+lines.children[x].width, y2 = y1;        
            var xK = this.children[0].position.x, yK = this.children[0].position.y;
            if (y1 > yK)
            {
                //копируем текстуру оригинала
                var copy_texture = lines.children[x]._texture.clone();

                var projection = new PIXI.Sprite(copy_texture);

                projection.position.x = (scena.H*(xK-x1)-xK*y1+x1*yK)/(yK-y1);
                projection.position.y = scena.H+this.myId*10;
                projection.width = (scena.H*(xK-x2)-xK*y2+x2*yK)/(yK-y2) - projection.position.x
                this.children[1].addChild(projection);
            }          
        } 
    }   

    //меню
    var displacementFolder = displacementFolderCamera.addFolder('camera'+scena.camerasId);
    cameraContainer.myId = scena.camerasId;
    scena.camerasId += 1;
    displacementFolder.add(camera, 'rotation', 0, 2*Math.PI).name("angle");
    displacementFolder.add(cameraContainer, 'project').name("project");
    displacementFolder.add(camera, 'isMatching').name("use for match"); 
    displacementFolder.add(camera, 'autoRender').name("use for auto render");    
    displacementFolder.add(camera, 'destroy').name("destroy");

}

scena.addLayer = function ()
{
    var x = Math.floor(Math.random() * MAXX);
    var y = Math.floor(Math.random() * VALIDY);

    var layer = new PIXI.Sprite(textureLine[Math.floor(Math.random() * textureLine.length)]);

    // enable the layer to be interactive... this will allow it to respond to mouse and touch events
    layer.interactive = true;

    // this button mode will mean the hand cursor appears when you roll over the layer with your mouse
    layer.buttonMode = true;

    // center the layer's anchor point
    layer.anchor.set(0);

    // setup events
    layer
        // events for drag start
        .on('mousedown', onDragStart)
        .on('touchstart', onDragStart)
        // events for drag end
        .on('mouseup', onDragEnd)
        .on('mouseupoutside', onDragEnd)
        .on('touchend', onDragEnd)
        .on('touchendoutside', onDragEnd)
        // events for drag move
        .on('mousemove', onDragMove)
        .on('touchmove', onDragMove);

    // move the sprite to its designated position
    layer.position.x = x;
    layer.position.y = y;

    layer.isPoint = false;

    //меню
    var displacementFolder = displacementFolderLayer.addFolder('line#'+scena.linesId);
    scena.linesId += 1;
    displacementFolder.add(layer, 'width', 4, MAXX).name("width");
    displacementFolder.add(layer, 'destroy').name("destroy");

    // add it to the stage
    lines.addChild(layer);
}

scena.addPoint = function ()
{
    var x = Math.floor(Math.random() * MAXX)
    var y = Math.floor(Math.random() * VALIDY);

    var point = new PIXI.Sprite(texturePoint[Math.floor(Math.random() * texturePoint.length)]);

    // enable the point to be interactive... this will allow it to respond to mouse and touch events
    point.interactive = true;

    // this button mode will mean the hand cursor appears when you roll over the point with your mouse
    point.buttonMode = true;

    // center the point's anchor point
    point.anchor.set(0);

    point.scale.set(3);

    // setup events
    point
        // events for drag start
        .on('mousedown', onDragStart)
        .on('touchstart', onDragStart)
        // events for drag end
        .on('mouseup', onDragEnd)
        .on('mouseupoutside', onDragEnd)
        .on('touchend', onDragEnd)
        .on('touchendoutside', onDragEnd)
        // events for drag move
        .on('mousemove', onDragMove)
        .on('touchmove', onDragMove);

    // move the sprite to its designated position
    point.position.x = x;
    point.position.y = y;

    point.isMatching = false;
    point.isPoint = true;

    displacementFolderPoint.add(point, 'isMatching').name('point#'+scena.pointsId);
    scena.pointsId += 1; 
  
    // add it to the stage
    lines.addChild(point);
}

scena.match = function ()
{
    for (i in cameras.children)
    {
        cameras.children[i].children[2].removeChildren();
    }

    var matchPoints = [], matchCamera = [];

    for (i in lines.children)
    {
        if (lines.children[i].isMatching)
            matchPoints.push(lines.children[i]);
    }
    for (i in cameras.children)
    {
        if (cameras.children[i].children[0].isMatching)
            matchCamera.push(cameras.children[i]);
    }
    if (matchPoints.length != 2) {
        alert("Need 2 keyPoint");
    }
    if (matchPoints.length != 2) {
        alert("Need 2 camera");
    }    
    var x1 = matchPoints[0].position.x, y1 = matchPoints[0].position.y;
    var x2 = matchPoints[1].position.x, y2 = matchPoints[1].position.y;       
    var xK1 = matchCamera[0].children[0].position.x, yK1 = matchCamera[0].children[0].position.y;
    var xK2 = matchCamera[1].children[0].position.x, yK2 = matchCamera[1].children[0].position.y;

    var x1K1 = (scena.H*(xK1-x1)-xK1*y1+x1*yK1)/(yK1-y1);
    var x1K2 = (scena.H*(xK2-x1)-xK2*y1+x1*yK2)/(yK2-y1);

    var x2K1 = (scena.H*(xK1-x2)-xK1*y2+x2*yK1)/(yK1-y2);
    var x2K2 = (scena.H*(xK2-x2)-xK2*y2+x2*yK2)/(yK2-y2); 
    var k = (x2K2-x1K2)/(x2K1-x1K1);
    var b = x2K2 - x2K1*k;                    

    console.log(x1K1, x2K1, x1K2, x2K2, k, b);

    for (x in lines.children)
    {
        //находим координаты концов слоев и камеры
        var x1 = lines.children[x].position.x, y1 = lines.children[x].position.y;
        var x2 = x1+lines.children[x].width, y2 = y1;        
        var xK = matchCamera[0].children[0].position.x, yK = matchCamera[0].children[0].position.y;
        if (y1 > yK)
        {
            //копируем текстуру оригинала
            var copy_texture = lines.children[x]._texture.clone();

            var projection = new PIXI.Sprite(copy_texture);

            projection.position.x = k*((scena.H*(xK-x1)-xK*y1+x1*yK)/(yK-y1))+b;
            projection.position.y = scena.H+50;
            console.log(projection.position.x);
            projection.width = k*((scena.H*(xK-x2)-xK*y2+x2*yK)/(yK-y2))+b-projection.position.x;
            matchCamera[0].children[2].addChild(projection);

        }          
    } 
    for (x in lines.children)
    {
        //находим координаты концов слоев и камеры
        var x1 = lines.children[x].position.x, y1 = lines.children[x].position.y;
        var x2 = x1+lines.children[x].width, y2 = y1;        
        var xK = matchCamera[1].children[0].position.x, yK = matchCamera[1].children[0].position.y;
        if (y1 > yK)
        {
            //копируем текстуру оригинала
            var copy_texture = lines.children[x]._texture.clone();

            var projection = new PIXI.Sprite(copy_texture);

            projection.position.x = (scena.H*(xK-x1)-xK*y1+x1*yK)/(yK-y1);
            projection.position.y = scena.H+60;
            console.log(projection.position.x);
            projection.width = (scena.H*(xK-x2)-xK*y2+x2*yK)/(yK-y2) - projection.position.x;
            matchCamera[1].children[2].addChild(projection);
        }          
    }    
}

var gui = new dat.GUI({});
var displacementFolderCamera = gui.addFolder('Cameras');
displacementFolderCamera.add(scena, 'addCamera').name("add camera"); 
var displacementFolderLayer = gui.addFolder('Layers');
displacementFolderLayer.add(scena, 'addLayer').name("add layer"); 
var displacementFolderPoint = gui.addFolder('Points');
displacementFolderPoint.add(scena, 'addPoint').name("add point"); 
gui.add(scena, 'match'); 
gui.add(scena, 'H', 0, VALIDY); 

var renderer = PIXI.autoDetectRenderer(window.innerWidth, window.innerHeight, {backgroundColor : 0xFFFFFF});
document.body.appendChild(renderer.view);

// create the root of the scene graph
var stage = new PIXI.Container();
var projections = new PIXI.Container();
var lines = new PIXI.Container();
var cameras = new PIXI.Container();
stage.addChild(projections);
stage.addChild(lines);
stage.addChild(cameras);

scena.addLayer(); scena.addLayer();

scena.addCamera(); scena.addCamera();

scena.addPoint(); scena.addPoint();

requestAnimationFrame( animate );

function animate() {
    for (i in cameras.children)
    {
        if (cameras.children[i].children[0].autoRender)
        {
            cameras.children[i].project();
        }
    }
    requestAnimationFrame(animate);

    // cameras[0].rotation += 0.1;
    // render the stage
    renderer.render(stage);

}

function onDragStart(event)
{
    // store a reference to the data
    // the reason for this is because of multitouch
    // we want to track the movement of this particular touch
    this.data = event.data;
    this.alpha = 0.5;
    this.dragging = true; 
//    stage.addChild(this);
}

function onDragEnd()
{
    this.alpha = 1;

    this.dragging = false;

    // set the interaction data to null
    this.data = null;
}

function onDragMove(mouseData)
{
    if (this.dragging)
    {
        var newPosition = this.data.getLocalPosition(this.parent);
        this.position.x += mouseData.data.originalEvent.movementX;
        this.position.y += mouseData.data.originalEvent.movementY;
        if (this.position.y > VALIDY)
        {
            this.position.y = VALIDY;
        }
    }
}

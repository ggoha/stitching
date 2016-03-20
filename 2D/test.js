var scena = {};
scena.H = 600;
scena.camerasId = 0; scena.linesId = 0; scena.pointsId = 0;
var MAXX = window.innerWidth, MAXY = window.innerHeight;
var matching = [];
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
    camera.position.y = Math.floor(Math.random() * MAXY);

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


    // add it the stage so we see it on our screens..
    cameraContainer.addChild(camera);

    var projection = new PIXI.Container();
    cameraContainer.addChild(projection);

    cameraContainer.project = function ()
    {
        console.log(this);
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
    var displacementFolder = gui.addFolder('camera'+scena.camerasId);
    cameraContainer.myId = scena.camerasId;
    scena.camerasId += 1;
    displacementFolder.add(camera, 'rotation', 0, 2*Math.PI).name("angle");
    displacementFolder.add(cameraContainer, 'project').name("project");
    displacementFolder.add(camera, 'destroy').name("destroy");
}

scena.addLayer = function ()
{
    var x = Math.floor(Math.random() * MAXX);
    var y = Math.floor(Math.random() * MAXY);
    // create our little bunny friend..
    var bunny = new PIXI.Sprite(textureLine[Math.floor(Math.random() * textureLine.length)]);

    // enable the bunny to be interactive... this will allow it to respond to mouse and touch events
    bunny.interactive = true;

    // this button mode will mean the hand cursor appears when you roll over the bunny with your mouse
    bunny.buttonMode = true;

    // center the bunny's anchor point
    bunny.anchor.set(0);

    // setup events
    bunny
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
    bunny.position.x = x;
    bunny.position.y = y;

    //меню
    var displacementFolder = gui.addFolder('line#'+scena.linesId);
    scena.linesId += 1;
    displacementFolder.add(bunny, 'width', 4, MAXX).name("width");
    displacementFolder.add(bunny, 'destroy').name("destroy");

    // add it to the stage
    lines.addChild(bunny);
}

scena.addPoint = function ()
{
    var x = Math.floor(Math.random() * MAXX)
    var y = Math.floor(Math.random() * MAXY);
    // create our little bunny friend..
    var bunny = new PIXI.Sprite(texturePoint[Math.floor(Math.random() * texturePoint.length)]);

    // enable the bunny to be interactive... this will allow it to respond to mouse and touch events
    bunny.interactive = true;

    // this button mode will mean the hand cursor appears when you roll over the bunny with your mouse
    bunny.buttonMode = true;

    // center the bunny's anchor point
    bunny.anchor.set(0);

    bunny.scale.set(3);

    // setup events
    bunny
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

    var displacementFolder = gui.addFolder('Point#'+scena.pointsId);
    scena.pointsId += 1;
    // move the sprite to its designated position
    bunny.position.x = x;
    bunny.position.y = y;

    // add it to the stage
    lines.addChild(bunny);
}

var gui = new dat.GUI({});
gui.add(scena, 'addCamera'); 
gui.add(scena, 'addLayer'); 
gui.add(scena, 'addPoint'); 
gui.add(scena, 'H', 0, MAXY); 

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

scena.addCamera();

requestAnimationFrame( animate );

function animate() {

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
    }
}

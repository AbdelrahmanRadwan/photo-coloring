//To check if the file is avalibe
$("#file-picker").change(function(){

    var input = document.getElementById('file-picker');

    for (var i=0; i<input.files.length; i++)
    {
    //koala.jpg, koala.JPG substring(index) lastIndexOf('a') koala.1.jpg
        var ext= input.files[i].name.substring(input.files[i].name.lastIndexOf('.')+1).toLowerCase()

        if ((ext == 'jpg') || (ext == 'png')||(ext == 'mp4'))
        {
            $("#msg").html('<div class="alert alert-success" role="alert"><strong>File is supported</strong> </div>')
        }
        else
        {
            $("#msg").html('<div class="alert alert-danger" role="alert">File is NOT supported</div>')
            document.getElementById("file-picker").value ="";
        }

    }

} );

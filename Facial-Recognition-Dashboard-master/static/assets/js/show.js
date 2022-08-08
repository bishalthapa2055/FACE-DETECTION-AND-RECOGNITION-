

$(document).ready(function()
{

    d = new Date();
    $("#myimg").attr("src", "http://localhost:5000/assets/img/test.jpg?"+d.getTime());


});


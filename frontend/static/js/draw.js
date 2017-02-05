var _timer;
var _sketch;

function clearImage() {
    var canvas = $('#sketch')[0];
    var ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    $('#sketch').sketch().actions = [];
}

function submitImage() {
    if (!_sketch.painting) return;
    var dataURL = $('#sketch')[0].toDataURL("image/png");
    $.post('/classify', {
      'imageb64': dataURL, 'limit': 5
    }).done(function(data) {
        drawTable(data.results);
    });
}

function drawTable(results) {
    $('#results').empty();
    var table = $('<table></table>');
    var head = $('<tr><th>Label</th><th>Score</th></tr>');
    table.append(head);
    for (i = 0; i < results.length; ++i) {
        var row = $('<tr></tr>');
        var label = $('<td></td>').text(results[i].label);
        var score = $('<td></td>').text(results[i].score);
        row.append(label);
        row.append(score);
        table.append(row);
    }
    $('#results').append(table);
}

$(function() {
    $('#sketch').sketch({
      defaultColor: 'white', 
      defaultSize: 15,
    });
    $('#start-sketch').click(function() {
      _timer = setInterval(submitImage, 100);
    });
    $('#clear-sketch').click(function () {
      clearImage();
    });
    $('#stop-sketch').click(function () {
      clearInterval(_timer);
      _timer = null;
    });
    _sketch = $('#sketch').data('sketch');
    console.log('Initialized!');
});

window.onload = function() {
	let outer_div = document.createElement("div");
	let inner_div = document.createElement("div");

	inner_div.innerHTML = "Awesome! See you soon!";
	inner_div.classList.add("user1", "msg_left_content");

	outer_div.appendChild(inner_div);
	outer_div.classList.add("msg", "msg_left");

	let container = document.getElementById("container");
	container.appendChild(outer_div);

	outer_div.scrollIntoView(false);
}

$(function(){
    $('#form-reply').submit(function(){
		$.ajax({
			url: '/reply',
			data: $('form').serialize(),
			type: 'POST',
			success: function(response){
				console.log(response);
			},
			error: function(error){
				console.log(error);
			}
		});
	});
});
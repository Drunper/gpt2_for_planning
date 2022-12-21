(define (problem logistics-test)
(:domain logistics)
(:objects
	apn8 apn6 - airplane
	cit4 cit1 - city
	apt5 apt3 - airport
	tru5 tru3 - truck
	obj77 obj21 obj23 obj33 obj55 obj99 - package
	pos11 pos12 pos33 pos55 pos21 pos23 - location
)
(:init
	(at apn8 apt5)
	(at apn6 apt5)
	(at obj77 pos23)
	(at obj21 pos33)
	(at obj23 pos33)
	(at obj33 pos33)
	(at obj55 pos11)
	(at obj99 pos11)
	(in-city apt5 cit1)
	(in-city apt3 cit4)
	(in-city pos11 cit4)
	(in-city pos12 cit4)
	(in-city pos33 cit1)
	(in-city pos55 cit4)
	(in-city pos21 cit4)
	(in-city pos23 cit1)
	(at tru3 pos55)
	(at tru5 pos33)
)
(:goal
	(and
		(at obj33 pos33)
		(at obj55 pos33)
))
)
